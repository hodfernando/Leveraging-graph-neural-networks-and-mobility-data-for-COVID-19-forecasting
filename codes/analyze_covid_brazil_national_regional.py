"""Análise COVID-19 Brasil: nacional, regional, persistência e conformal."""
import warnings,pickle
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
LAG=14; HORIZON=14; TRAIN_RATIO=.8; N_EVENTS=4; EVENT_WINDOW=21; SURGE_WINDOW=14; MIN_RELATIVE_RISE=.20; MIN_EVENT_GAP=35; TOP_N=5
REGIONS={1:"Norte",2:"Nordeste",3:"Sudeste",4:"Sul",5:"Centro-Oeste"}; MODELS=["GCRN","GCLSTM","LSTM"]

def root():
    p=Path(__file__).resolve().parent.parent; d=Path(str(p).replace("/media/work/","/media/data/",1)); return d if (d/"results_daily/Brazil").exists() else p

def outdir(r):
    o=r/"results_daily/results_analysis/Brazil/covid_lags_14_out_14"; (o/"tables").mkdir(parents=True,exist_ok=True); (o/"figures").mkdir(parents=True,exist_ok=True); return o

def load_preds(r):
    base=r/"results_daily/Brazil/regression/grafo_original/backbone_alpha_1/k_1_hc_256"; z={}
    for m in MODELS:
        d=base/m/"lags_14_out_14"; real=sorted(d.glob("y_real_no_norm_rep_*.npy")); pairs=[]
        for x in real:
            y=d/x.name.replace("y_real_no_norm_rep_","y_pred_no_norm_rep_")
            if y.exists(): pairs.append((x,y))
        if not pairs: raise FileNotFoundError(f"Previsões ausentes: {d}")
        z[m]={"real":[np.load(a,mmap_mode="r") for a,b in pairs],"pred":[np.load(b,mmap_mode="r") for a,b in pairs]}; print(f"[INFO] {m}: {len(pairs)} repetições, shape={z[m]['real'][0].shape}")
    return z

def dates(r,n):
    import polars as pl
    f=r/"raw_data/Brazil/cases-brazil-cities-time_changesOnly.csv.gz"; d=pd.to_datetime(pl.read_csv(f,columns=["date"])["date"].unique().sort().to_numpy()); ns=len(d)-LAG-HORIZON+1; cut=int(ns*TRAIN_RATIO); idx=np.clip(cut+np.arange(n)[:,None]+LAG+np.arange(HORIZON)[None,:],0,len(d)-1); return d.to_numpy()[idx],d

def codes(r):
    p=r/"pre_processed/Brazil/codigos_municipios_grafo_original_backbone_0.01.pkl"; 
    with open(p,"rb") as f:return np.asarray(pickle.load(f))

def cases(r,ibge,d):
    import polars as pl
    x=pl.read_csv(r/"raw_data/Brazil/cases-brazil-cities-time_changesOnly.csv.gz",columns=["date","ibgeID","newCases"]); dt={pd.Timestamp(v):i for i,v in enumerate(pd.to_datetime(d))}; ci={int(v):i for i,v in enumerate(ibge)}; a=np.zeros((len(d),len(ibge)),np.float32)
    for row in x.iter_rows(named=True):
        k=pd.Timestamp(row["date"]); c=int(row["ibgeID"])
        if k in dt and c in ci:a[dt[k],ci[c]]=max(float(row["newCases"] or 0),0)
    return a

def build_persistence(a,n):
    ns=len(a)-LAG-HORIZON+1; cut=int(ns*TRAIN_RATIO); p=np.zeros((n,a.shape[1],HORIZON),np.float32)
    for s in range(n):p[s]=a[cut+s+LAG-1,:,None]
    return p

def mean(a):return np.mean(np.stack([np.asarray(x) for x in a]),axis=0)

def event_detection(y,td):
    total=y.sum(1)[:,0]; smooth=pd.Series(total).rolling(7,center=True,min_periods=1).mean().to_numpy(); c=[]
    for i in range(1,len(smooth)-1):
        if smooth[i]>=smooth[i-1] and smooth[i]>=smooth[i+1]:
            prev=smooth[max(0,i-SURGE_WINDOW)]; rise=smooth[i]-prev; rel=rise/max(prev,1)
            if rise>0 and rel>=MIN_RELATIVE_RISE:c.append((i,smooth[i],prev,rise,rel))
    c.sort(key=lambda q:q[3],reverse=True); sel=[]
    for q in c:
        if all(abs(q[0]-s["snapshot"])>=MIN_EVENT_GAP for s in sel):sel.append({"snapshot":q[0],"peak_total_cases":q[1],"pre_peak_total":q[2],"absolute_rise":q[3],"relative_rise":q[4],"date":td[q[0],0]})
        if len(sel)==N_EVENTS:break
    sel.sort(key=lambda q:q["snapshot"]); ev=pd.DataFrame(sel)
    if not ev.empty:ev.insert(0,"event",range(1,len(ev)+1))
    return ev,pd.DataFrame({"snapshot":range(len(total)),"date":td[:,0],"national_total":total,"national_smooth7":smooth})

def windows(ev,n,ser):
    q=[]
    for _,e in ev.iterrows():
        p=int(e.snapshot)
        for s in range(max(0,p-EVENT_WINDOW),min(n,p+EVENT_WINDOW+1)):
            d=s-p;q.append({"event":int(e.event),"snapshot":s,"days_from_peak":d,"phase":"surge" if d<-7 else ("peak" if d<=7 else "decline"),"date":ser.loc[s,"date"]})
    return pd.DataFrame(q).drop_duplicates(["event","snapshot"])

def reg(c):
    try:return REGIONS.get(int(str(int(c)).zfill(7)[0]),"Desconhecida")
    except:return "Desconhecida"

def metric_rows(m,real,pred,w,regions):
    rows=[]
    for _,x in w.iterrows():
        e=pred[int(x.snapshot)]-real[int(x.snapshot)]
        for rr in ["Brasil","Norte","Nordeste","Sudeste","Sul","Centro-Oeste"]:
            mask=np.ones(len(regions),bool) if rr=="Brasil" else regions==rr
            if mask.sum():
                a=e.sum(0) if rr=="Brasil" else e[mask].sum(0); rows.append({"model":m,"event":int(x.event),"phase":x.phase,"snapshot":int(x.snapshot),"date":x.date,"region":rr,"RMSE":float(np.sqrt(np.mean(a*a))),"MAE":float(np.mean(np.abs(a)))})
    return rows

def metrics(data,pers,w,regions,o):
    rows=[]; real_ref=mean(data["GCRN"]["real"])
    for m in MODELS:rows+=metric_rows(m,mean(data[m]["real"]),mean(data[m]["pred"]),w,regions)
    rows+=metric_rows("Persistence",real_ref,pers,w,regions); detail=pd.DataFrame(rows); summary=detail.groupby(["model","region","phase"],as_index=False)[["RMSE","MAE"]].mean(); detail.to_csv(o/"tables/event_metrics_national_regional_by_snapshot.csv",index=False); summary.to_csv(o/"tables/event_metrics_national_regional_summary.csv",index=False); return detail,summary

def network(r):
    d=r/"raw_data/Brazil/in"; frames=[]
    for m in ["betweenness","strength","degree","closeness"]:
        p=d/f"{m}.csv"
        if p.exists():
            x=pd.read_csv(p,sep=";",header=None,names=["city","ibgeID","value"]);x["ibgeID"]=pd.to_numeric(x.ibgeID,errors="coerce").astype("Int64");frames.append(x.rename(columns={"value":m})[["ibgeID",m,"city"]])
    z=frames[0]
    for x in frames[1:]:z=z.merge(x.drop(columns="city"),on="ibgeID",how="outer")
    return z

def top(net,o):
    a=[]
    for m in ["betweenness","strength","degree","closeness"]:
        x=net.nlargest(TOP_N,m)[["ibgeID","city",m]].rename(columns={m:"value"});x["metric"]=m;a.append(x)
    allx=pd.concat(a);s=allx.groupby(["ibgeID","city"]).agg(n_metrics_in_top=("metric","count"),mean_value=("value","mean")).reset_index().sort_values(["n_metrics_in_top","mean_value"],ascending=False);s["rank"]=range(1,len(s)+1);s.head(TOP_N).to_csv(o/"tables/top5_important_cities.csv",index=False);allx.to_csv(o/"tables/top5_by_network_metric.csv",index=False);return s.head(TOP_N)

def figures(ser,summary,o):
    fig,ax=plt.subplots(figsize=(14,6));ax.plot(ser.date,ser.national_total,color="lightgray",label="Total diário");ax.plot(ser.date,ser.national_smooth7,color="black",lw=2,label="Média móvel 7 dias");ax.set_title("Casos nacionais e eventos aceitos — lags_14_out_14");ax.set_ylabel("Casos");ax.legend();fig.tight_layout();fig.savefig(o/"figures/national_series_events.png",dpi=300);plt.close(fig)
    for regs,name in [(["Norte","Nordeste"],"regional_event_metrics_norte_nordeste.png"),(["Sudeste","Sul","Centro-Oeste"],"regional_event_metrics_sudeste_sul_centrooeste.png")]:
        fig,axes=plt.subplots(1,len(regs),figsize=(8*len(regs),7),squeeze=False);axes=axes[0]
        for ax,rr in zip(axes,regs):summary[summary.region==rr].pivot(index="phase",columns="model",values="RMSE").reindex(["surge","peak","decline"]).plot(kind="bar",ax=ax);ax.set_title(rr);ax.set_xlabel("Fase");ax.set_ylabel("RMSE");ax.tick_params(axis="x",rotation=0)
        fig.suptitle("RMSE regional por fase — persistência corrigida",weight="bold");fig.tight_layout();fig.savefig(o/"figures"/name,dpi=300,bbox_inches="tight");plt.close(fig)

def conformal(data,pers,w,o):
    rows=[]
    for m in MODELS+["Persistence"]:
        real=mean(data["GCRN"]["real"]);pred=pers if m=="Persistence" else mean(data[m]["pred"]);scores=np.abs(real-pred);q=np.quantile(scores,.95,axis=0);lower=pred-q;upper=pred+q
        for _,x in w.iterrows():
            s=int(x.snapshot);inside=(real[s]>=lower[s])&(real[s]<=upper[s]);rows.append({"model":m,"event":int(x.event),"phase":x.phase,"coverage_95":inside.mean(),"interval_width":np.mean(upper[s]-lower[s]),"quantile_95":np.mean(q)})
    df=pd.DataFrame(rows);df.to_csv(o/"tables/conformal_event_metrics.csv",index=False);summary=df.groupby(["model","phase"],as_index=False)[["coverage_95","interval_width","quantile_95"]].mean();summary.to_csv(o/"tables/conformal_summary.csv",index=False);return summary

def report(o,ev,summary,top,conf):
    text="# Análise COVID-19 Brasil — lags_14_out_14\n\n## Eventos\n\n"+ev.to_markdown(index=False)+"\n\n## Métricas\n\n"+summary.to_markdown(index=False)+"\n\n## Conformal Prediction\n\n"+conf.to_markdown(index=False)+"\n\n## Cidades importantes\n\n"+top.to_markdown(index=False)+"\n"; (o/"report_national_regional.md").write_text(text,encoding="utf-8")

def main():
    r=root();o=outdir(r);data=load_preds(r);n=data["GCRN"]["real"][0].shape[0];td,all_dates=dates(r,n);ibge=codes(r);a=cases(r,ibge,all_dates);pers=build_persistence(a,n);real=mean(data["GCRN"]["real"]);ev,ser=event_detection(real,td);ev.to_csv(o/"tables/national_events.csv",index=False);ser.to_csv(o/"tables/national_series.csv",index=False);w=windows(ev,n,ser);w.to_csv(o/"tables/national_event_windows.csv",index=False);regions=np.array([reg(x) for x in ibge]);detail,summary=metrics(data,pers,w,regions,o);net=network(r);selected=top(net,o);figures(ser,summary,o);conf=conformal(data,pers,w,o);report(o,ev,summary,selected,conf);print(f"[INFO] Relatórios, tabelas e figuras sobrescritos em {o}")

if __name__=="__main__":main()