from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score

NAME = "NF-CSE-CIC-IDS2018-v3"
SEED = 24
TS = "FLOW_START_MILLISECONDS"
RAW = Path(f"data/{NAME}/{NAME}.csv")
CACHE = Path(f"data/pyg_graph_data/{NAME}_0_2")
SCORES = Path("results/failure_analysis/metrics/audit_scores_20pct_shuffled.csv")
CKPT = Path(f"checkpoints/GraphIDS_{NAME}_{SEED}.ckpt")
OUT = SCORES.with_name("audit_aligned_20pct_shuffled.csv")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def align_metadata():
    import torch
    require(not OUT.exists(), f"Output already exists: {OUT}. Keep it; do not overwrite.")
    for path in (RAW, SCORES, CKPT, CACHE / "test.pt", CACHE / "scaler.pkl", CACHE / ".seed"):
        require(path.is_file(), f"Missing required file: {path}")
    require(int((CACHE / ".seed").read_text().strip()) == SEED, "Cache seed is not 24.")
    scores = pd.read_csv(SCORES)
    g = torch.load(CACHE / "test.pt", map_location="cpu", weights_only=False)[0]
    n = g.edge_index.shape[1]
    require(n == 402311 and len(scores) == n, "Unexpected test or score count.")
    require(np.array_equal(scores.flow_idx.to_numpy(), np.arange(n)), "Score IDs must cover test edges in order.")
    require(np.isfinite(scores.recon_error.to_numpy()).all(), "Nonfinite scores.")
    graph_labels = g.edge_labels.cpu().numpy()
    require(np.array_equal(scores.label.to_numpy(), graph_labels), "Scores disagree with graph labels.")
    require(int((graph_labels == 1).sum()) == 52018, "Unexpected attack count.")

    print("Reconstructing the original 20% sample and stratified split...", flush=True)
    light = pd.read_csv(RAW, usecols=[TS, "Attack", "Label"])
    # Preserve the original CSV index: this is the raw row identifier.
    light = light.groupby(by="Attack").sample(frac=0.2, random_state=SEED)
    light[TS] = light[TS].replace([np.inf, -np.inf], np.nan).fillna(0)
    _, vt = train_test_split(light, test_size=0.2, random_state=SEED, stratify=light["Attack"])
    _, test = train_test_split(vt, test_size=0.5, random_state=SEED, stratify=vt["Attack"])
    test = test.sort_values(by=TS)
    row_ids = test.index.to_numpy()
    require(len(test) == n, "Reconstructed split has the wrong size.")
    require(np.array_equal(test.Label.to_numpy(), graph_labels), "Reconstructed labels do not match cached graph.")
    require(len(np.unique(row_ids)) == n, "Repeated raw row identifiers.")
    del light, vt

    print("Reading the raw CSV in chunks to recover the exact selected rows...", flush=True)
    pieces = []
    for chunk in pd.read_csv(RAW, chunksize=100000):
        selected = chunk.loc[chunk.index.isin(row_ids)]
        if not selected.empty:
            pieces.append(selected.copy())
    require(bool(pieces), "No raw metadata recovered.")
    meta = pd.concat(pieces).loc[row_ids].copy()
    require(len(meta) == n and meta.index.is_unique, "Incomplete or duplicate raw metadata.")
    for col in meta.columns:
        if col not in ("Attack", "Label"):
            meta[col] = meta[col].replace([np.inf, -np.inf], np.nan).fillna(0)
    require(np.array_equal(meta.Label.to_numpy(), graph_labels), "Raw labels do not match graph labels.")
    require(np.array_equal(meta[TS].to_numpy(), test[TS].to_numpy()), "Timestamp reconstruction differs.")

    with open(CACHE / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    excluded = {"Attack", "Label", "IPV4_SRC_ADDR", "IPV4_DST_ADDR", TS, "FLOW_END_MILLISECONDS"}
    features = [c for c in meta.columns if c not in excluded]
    if hasattr(scaler, "feature_names_in_"):
        require(features == list(scaler.feature_names_in_), "Feature names/order differ from saved scaler.")
    expected = np.clip(scaler.transform(meta[features]), -10, 10).astype(np.float32)
    actual = g.edge_attr.cpu().numpy()
    require(expected.shape == actual.shape, "Feature shape differs from cached graph.")
    require(np.allclose(expected, actual, rtol=1e-5, atol=1e-6), "Raw features/scaler do not reproduce cached graph. Stop.")
    print("PASS: all edge features and binary labels match the cached graph.", flush=True)

    ips = pd.concat([meta.IPV4_SRC_ADDR, meta.IPV4_DST_ADDR], ignore_index=True)
    node_ids = g.edge_index.cpu().numpy().reshape(-1)
    pairs = pd.DataFrame({"ip": ips.to_numpy(), "node_id": node_ids}).drop_duplicates()
    require(pairs.groupby("ip").node_id.nunique().max() == 1, "One IP maps to multiple node IDs.")
    require(pairs.groupby("node_id").ip.nunique().max() == 1, "One node ID maps to multiple IPs.")
    print("PASS: IP/node mapping is one-to-one.")

    out = scores.copy()
    out["raw_row_id"] = row_ids
    for source, target in [(TS, "timestamp_ms"), ("IPV4_SRC_ADDR", "src_ip"),
                           ("IPV4_DST_ADDR", "dst_ip"), ("Attack", "attack_type")]:
        out[target] = meta[source].to_numpy()
    out["total_bytes"] = (meta.IN_BYTES + meta.OUT_BYTES).to_numpy()
    out["total_pkts"] = (meta.IN_PKTS + meta.OUT_PKTS).to_numpy()
    out["duration_ms"] = meta.FLOW_DURATION_MILLISECONDS.to_numpy()
    out["protocol"] = meta.PROTOCOL.to_numpy()
    tau = float(torch.load(CKPT, map_location="cpu", weights_only=False)["threshold"])
    require(np.isfinite(tau) and tau > 0, "Invalid checkpoint threshold.")
    pred = out.recon_error.to_numpy() > tau
    print("Threshold:", format(tau, ".17g"))
    print("Macro F1:", f1_score(out.label, pred, average="macro"))
    print("Attack AP:", average_precision_score(out.label, out.recon_error))
    print("Confusion matrix [benign, attack]:", confusion_matrix(out.label, pred, labels=[0, 1]).tolist())
    print("Attack category vs binary label:")
    print(pd.crosstab(out.attack_type, out.label).to_string())
    print("Timestamp range UTC:", pd.to_datetime(out.timestamp_ms.min(), unit="ms", utc=True),
          "to", pd.to_datetime(out.timestamp_ms.max(), unit="ms", utc=True))
    print("Timestamps monotone:", out.timestamp_ms.is_monotonic_increasing)
    with OUT.open("x") as f:
        out.to_csv(f, index=False)
    print("SUCCESS. Saved:", OUT)

def digest(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def extract_scores():
    import random
    import sys
    import torch
    from torch.utils.data import DataLoader
    from torch_geometric.loader import LinkNeighborLoader
    sys.path.insert(0, str(Path.cwd()))
    from models.graphids import GraphIDS
    from utils.dataloaders import NetFlowDataset, SequentialDataset, collate_fn
    from utils.trainers import calculate_errors

    for filename in ('train.pt', 'val.pt', 'test.pt', 'scaler.pkl', '.seed'):
        require((CACHE / filename).is_file(), f'Missing cache file: {CACHE / filename}')
    require(int((CACHE / '.seed').read_text().strip()) == SEED, 'Unexpected cache seed.')
    require(CKPT.is_file(), f'Missing checkpoint: {CKPT}')
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    ds = NetFlowDataset(name=NAME, data_dir='data', seed=SEED,
                        fraction=0.2, split_mode='stratified', force_reload=False)
    g = ds.test_graph
    require(g.edge_index.shape[1] == 402311, 'Unexpected test set size.')
    require(int((g.edge_labels == 1).sum()) == 52018, 'Unexpected attack count.')
    model = GraphIDS(ndim_in=ds.num_node_features, edim_in=ds.num_edge_features,
                     edim_out=64, embed_dim=32, num_heads=4, num_layers=1,
                     window_size=512, dropout=0.5, ae_dropout=0.2,
                     positional_encoding=None, agg_type='mean', mask_ratio=0.15,
                     ae_feedforward_dim=256)
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    model.load_state_dict(ck['model_state_dict'])
    model.eval()
    tau = float(ck['threshold'])
    require(np.isfinite(tau) and tau > 0, 'Invalid checkpoint threshold.')
    loader = LinkNeighborLoader(data=g, num_neighbors=[32768],
                                edge_label_index=g.edge_index, edge_label=g.edge_labels,
                                batch_size=16384, shuffle=True, num_workers=0, drop_last=False)
    ids, errors, labels = [], [], []
    with torch.inference_mode():
        for b in loader:
            b.batch_edge_couples = b.edge_label_index.t()
            emb = model.encoder(b.edge_index, b.edge_attr, b.batch_edge_couples, b.num_nodes)
            ae = DataLoader(SequentialDataset(emb, window=512, step=512, device='cpu'),
                            batch_size=64, collate_fn=collate_fn)
            e = torch.cat([calculate_errors(model.transformer(x, mask), x, mask) for x, mask in ae])
            require(len(e) == len(b.edge_label), 'Score count mismatch; no padding or trimming allowed.')
            ids.append(b.input_id.cpu())
            errors.append(e.cpu())
            labels.append(b.edge_label.cpu())
    ids = torch.cat(ids).numpy()
    order = np.argsort(ids)
    require(np.array_equal(ids[order], np.arange(g.edge_index.shape[1])), 'Missing/repeated target edge IDs.')
    lab = torch.cat(labels).numpy()[order]
    err = torch.cat(errors).numpy()[order]
    require(np.array_equal(lab, g.edge_labels.cpu().numpy()), 'Target edge labels differ.')
    require(np.isfinite(err).all(), 'Nonfinite scores.')
    pd.DataFrame({'flow_idx': np.arange(len(err)), 'label': lab, 'recon_error': err}).to_csv(SCORES, index=False)
    return tau

def analyze(path, outdir, tau):
    from sklearn.metrics import precision_score, recall_score
    d = pd.read_csv(path)
    cols = {'flow_idx','raw_row_id','label','recon_error','timestamp_ms','attack_type',
            'src_ip','dst_ip','total_bytes','total_pkts','duration_ms','protocol'}
    require(cols.issubset(d.columns), f'Missing columns: {sorted(cols-set(d.columns))}')
    require(len(d) > 0 and not d[list(cols)].isna().any().any(), 'Empty data or missing values.')
    require(np.array_equal(d.flow_idx.to_numpy(), np.arange(len(d))), 'Flow IDs must be complete and ordered.')
    require(d.raw_row_id.is_unique, 'Repeated raw row IDs.')
    require(set(d.label.unique()) == {0,1}, 'Both binary classes must be present.')
    require(np.isfinite(d.recon_error).all() and (d.recon_error >= 0).all(), 'Invalid scores.')
    require(np.isfinite(tau) and tau > 0, 'Invalid threshold.')
    d['timestamp'] = pd.to_datetime(d.timestamp_ms, unit='ms', utc=True)
    d['hour_utc'] = d.timestamp.dt.floor('h')
    d['date_utc'] = d.timestamp.dt.strftime('%Y-%m-%d')
    d['detected'] = d.recon_error > tau
    a = d[d.label == 1].copy()
    b = d[d.label == 0].copy()
    missed = a[~a.detected]
    ratio = missed.recon_error / tau
    cm = confusion_matrix(d.label, d.detected, labels=[0,1])
    category = a.groupby('attack_type').detected.agg(total_flows='size', detected='sum')
    category['missed'] = category.total_flows-category.detected
    category['detection_rate_pct'] = 100*category.detected/category.total_flows
    category.to_csv(outdir/'detection_by_attack_type.csv')
    pd.crosstab(d.attack_type,d.label).to_csv(outdir/'label_consistency.csv')
    hourly = d.groupby('hour_utc').size().rename('all_flows').to_frame()
    ah = a.groupby('hour_utc').detected.agg(attack_flows='size', detected_attacks='sum')
    bh = b.groupby('hour_utc').detected.agg(benign_flows='size', false_positives='sum')
    hourly = hourly.join(ah).join(bh)
    for c in ['attack_flows','detected_attacks','benign_flows','false_positives']:
        hourly[c] = hourly[c].fillna(0).astype(int)
    hourly['missed_attacks'] = hourly.attack_flows-hourly.detected_attacks
    hourly['attack_detection_pct'] = 100*hourly.detected_attacks/hourly.attack_flows.replace(0,np.nan)
    hourly['benign_fpr_pct'] = 100*hourly.false_positives/hourly.benign_flows.replace(0,np.nan)
    ih = a[a.attack_type.str.lower().str.contains('infilt')].groupby('hour_utc').detected.agg(infiltration_flows='size', detected_infiltration='sum')
    hourly = hourly.join(ih)
    hourly['infiltration_detection_pct'] = 100*hourly.detected_infiltration/hourly.infiltration_flows
    hourly.to_csv(outdir/'hourly_results.csv')
    daily = d.groupby(['date_utc','label']).recon_error.agg(['size','median',lambda x:x.quantile(.9),lambda x:x.quantile(.99)])
    daily.columns=['flows','error_median','error_p90','error_p99']
    daily.to_csv(outdir/'daily_score_distribution.csv')
    stats = []
    for hour, group in b.groupby('hour_utc'):
        r = group.recon_error/tau
        stats.append({'hour_utc':hour,'benign_flows':len(group),'median_error_ratio':r.median(),
                      'p90_error_ratio':r.quantile(.9),'p99_error_ratio':r.quantile(.99),
                      'benign_fpr_pct':100*(r>1).mean()})
    pd.DataFrame(stats).to_csv(outdir/'hourly_benign_scores.csv',index=False)
    rows=[]
    for mult in [.1,.25,.5,.75,1,1.5,2,3,5,10]:
        p = d.recon_error>tau*mult
        tn,fp,fn,tp = confusion_matrix(d.label,p,labels=[0,1]).ravel()
        rows.append(dict(multiplier=mult,threshold=tau*mult,macro_f1=f1_score(d.label,p,average='macro',zero_division=0),
                         macro_precision=precision_score(d.label,p,average='macro',zero_division=0),
                         macro_recall=recall_score(d.label,p,average='macro',zero_division=0),
                         true_negatives=int(tn),false_positives=int(fp),false_negatives=int(fn),true_positives=int(tp),
                         attack_detection_pct=100*tp/(tp+fn),benign_fpr_pct=100*fp/(fp+tn)))
    sweep=pd.DataFrame(rows)
    sweep.to_csv(outdir/'threshold_sensitivity.csv',index=False)
    bins=pd.cut(ratio,[-np.inf,.1,.5,1],right=True)
    bins.value_counts(sort=False).rename('missed_flows').to_csv(outdir/'missed_ratio_bins.csv')
    pd.DataFrame({'flow_idx':missed.flow_idx,'raw_row_id':missed.raw_row_id,'error_ratio':ratio}).to_csv(outdir/'missed_error_ratios.csv',index=False)
    traffic=[]
    for cat,group in a.groupby('attack_type'):
        for detected,sub in group.groupby('detected'):
            for feat in ['total_bytes','total_pkts','duration_ms']:
                traffic.append(dict(attack_type=cat,detected=bool(detected),feature=feat,flows=len(sub),median=sub[feat].median(),mean=sub[feat].mean()))
    pd.DataFrame(traffic).to_csv(outdir/'traffic_by_category_and_detection.csv',index=False)
    summary=dict(rows=len(d),attack_flows=len(a),benign_flows=len(b),threshold=tau,
                 macro_f1=float(f1_score(d.label,d.detected,average='macro')),
                 attack_average_precision=float(average_precision_score(d.label,d.recon_error)),
                 confusion_matrix_benign_attack=cm.tolist(),missed_attacks=len(missed),
                 near_miss_definition='0.5 < error/threshold <= 1 (prediction uses strict >)',
                 near_misses=int((ratio>.5).sum()),deep_miss_definition='error/threshold < 0.1',
                 deep_misses=int((ratio<.1).sum()),mean_missed_ratio=float(ratio.mean()) if len(ratio) else None,
                 attack_hours=int((hourly.attack_flows>0).sum()),hours_with_misses=int((hourly.missed_attacks>0).sum()),
                 top_two_hours_missed_share=float(hourly.missed_attacks.nlargest(2).sum()/len(missed)) if len(missed) else None,
                 note='Single checkpoint and inference pass. Time and category associations are descriptive, not causal. No adaptive threshold is evaluated.')
    import json
    (outdir/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
    print('\nCATEGORY RESULTS\n'+category.to_string())
    print('\nTHRESHOLD SWEEP\n'+sweep.to_string(index=False))
    make_figures(d,a,b,ratio,hourly,sweep,tau,outdir)


def make_figures(d,a,b,ratio,hourly,sweep,tau,outdir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figs=outdir/'figures';figs.mkdir()
    def save(fig,name):
        fig.tight_layout()
        fig.savefig(figs/(name+'.pdf'),bbox_inches='tight')
        fig.savefig(figs/(name+'.png'),dpi=200,bbox_inches='tight')
        plt.close(fig)
    fig,ax=plt.subplots(2,1,figsize=(9,6),sharex=True)
    ax[0].plot(hourly.index,hourly.attack_detection_pct,'o',ms=3,label='All attacks')
    ax[0].plot(hourly.index,hourly.infiltration_detection_pct,'x',ms=4,label='Infiltration')
    ax[0].set(ylabel='Detection (%)',ylim=(-3,103));ax[0].legend()
    ax[1].plot(hourly.index,hourly.benign_fpr_pct,'o',ms=3)
    ax[1].set(ylabel='Benign false positives (%)',xlabel='UTC calendar hour')
    fig.autofmt_xdate();save(fig,'temporal_detection')
    fig,ax=plt.subplots(1,2,figsize=(8,3))
    edges=np.linspace(0,2,81)
    ax[0].hist(b.recon_error/tau,bins=edges,density=True,histtype='step',label='Benign')
    ax[0].hist(a[a.attack_type.str.lower().str.contains('infilt')].recon_error/tau,bins=edges,density=True,histtype='step',label='Infiltration')
    ax[0].axvline(1,color='black',ls='--');ax[0].set(xlabel='Error / threshold',ylabel='Density (restricted to 0–2)');ax[0].legend()
    ax[1].hist(ratio,bins=np.linspace(0,1,41));ax[1].axvline(.5,color='black',ls='--')
    ax[1].set(xlabel='Missed error / threshold',ylabel='Missed attacks');save(fig,'score_diagnostics')
    fig,ax=plt.subplots(1,2,figsize=(8,3))
    ax[0].plot(sweep.multiplier,sweep.macro_f1,'o-');ax[0].set(xscale='log',xlabel='Threshold multiplier',ylabel='Macro F1')
    ax[1].plot(sweep.multiplier,sweep.attack_detection_pct,'o-',label='Attack detection')
    ax[1].plot(sweep.multiplier,sweep.benign_fpr_pct,'s-',label='Benign FPR')
    ax[1].set(xscale='log',xlabel='Threshold multiplier',ylabel='Percent');ax[1].legend();save(fig,'threshold_tradeoff')


def cli():
    import argparse,json,platform,sys,importlib.metadata
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['analyze','checkpoint'])
    parser.add_argument('--input',type=Path,help='Verified aligned CSV; required in analyze mode')
    parser.add_argument('--threshold',type=float,default=3.4440622584952507e-06,
                        help='Saved seed-24 threshold for CSV mode; never optimized on test data')
    parser.add_argument('--out',type=Path,required=True,help='NEW directory; existing directories are refused')
    args=parser.parse_args()
    if args.mode=='analyze':
        require(args.input is not None and args.input.is_file(),'Provide --input with the verified aligned CSV.')
    require(not args.out.exists(),'Output directory already exists; choose a new name.')
    args.out.mkdir(parents=True)
    tau=args.threshold
    global SCORES,OUT
    if args.mode=='checkpoint':
        SCORES=args.out/'scores.csv';OUT=args.out/'aligned_scores.csv'
        tau=extract_scores()
        align_metadata()
        path=OUT
    else:
        path=args.input
    analyze(path,args.out,tau)
    versions={}
    for pkg in ['numpy','pandas','scikit-learn','matplotlib','torch','torch-geometric']:
        try: versions[pkg]=importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError: versions[pkg]='not installed'
    manifest=dict(command=sys.argv,mode=args.mode,input_path=str(path.resolve()),input_sha256=digest(path),
                  script_sha256=digest(Path(__file__)),threshold=tau,python=platform.python_version(),versions=versions,
                  provenance='CSV mode relies on upstream alignment checks; it cannot independently revalidate the raw graph.')
    if args.mode=='checkpoint':
        manifest.update(checkpoint_sha256=digest(CKPT),scaler_sha256=digest(CACHE/'scaler.pkl'),
                        test_graph_sha256=digest(CACHE/'test.pt'),raw_csv_sha256=digest(RAW),
                        seed=SEED,fraction=.2,shuffle=True,num_workers=0,device='cpu')
    (args.out/'provenance.json').write_text(json.dumps(manifest,indent=2))
    print('\nSUCCESS. Outputs:',args.out)


if __name__=='__main__':
    cli()
