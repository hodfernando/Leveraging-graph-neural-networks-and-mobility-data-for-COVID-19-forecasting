# coding: utf-8
"""Calcula erro relativo e métricas de incidência por 100 mil habitantes.

Configuração: COVID-19 Brasil, lags=14, horizonte=14.
Usa os arrays de previsões e as mesmas janelas nacionais de eventos já geradas.
"""
import warnings,pickle
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
LAG=14; HORIZON=14; TRAIN_RATIO=.8
MODELS=["GCRN","GCLSTM","LSTM"]
REGIONS={1:"Norte",2:"Nordeste",3:"Sudeste",4:"Sul",5:"Centro-Oeste"}
REGION_ORDER=["Brasil","Norte","Nordeste","Sudeste","Sul","Centro-Oeste"]
PHASE_ORDER=["surge","peak","decline"]


def root():
    p=Path(__file__).resolve().parent.parent; d=Path(str(p).replace("/media/work/","/media/data/",1)); return d if (d/"results_daily/Brazil").exists() else p

def output(root):
    o=root/"results_daily/results_analysis/Brazil/covid_lags_14_out_14"; (o/"tables").mkdir(parents=True,exist_ok=True);(o/"figures").mkdir(parents=True,exist_ok=True);return o

def load_arrays(root):
    base=root/"results_daily/Brazil/regression/grafo_original/backbone_alpha_1/k_1_hc_256"; result={}
    for model in MODELS:
        d=base/model/"lags_14_out_14"; rf=sorted(d.glob("y_real_no_norm_rep_*.npy")); pairs=[]
        for x in rf:
            y=d/x.name.replace("y_real_no_norm_rep_","y_pred_no_norm_rep_")
            if y.exists(): pairs.append((x,y))
        if not pairs: raise FileNotFoundError(f"Previsões ausentes: {d}")
        result[model]={"real":[np.load(a,mmap_mode="r") for a,b in pairs],"pred":[np.load(b,mmap_mode="r") for a,b in pairs]}
        print(f"[INFO] {model}: {len(pairs)} repetições, shape={result[model]['real'][0].shape}")
    return result

def graph_codes(root):
    p=root/"pre_processed/Brazil/codigos_municipios_grafo_original_backbone_0.01.pkl"
    with open(p,"rb") as f:return np.asarray(pickle.load(f))

def region(code):
    try:return REGIONS.get(int(str(int(code)).zfill(7)[0]),"Desconhecida")
    except:return "Desconhecida"

def load_population(root,codes):
    import polars as pl
    candidates=list((root/"raw_data/Brazil").glob("*Populacao*.xlsx"))+list((root/"raw_data/Brazil").glob("*populacao*.xlsx"))
    if not candidates: raise FileNotFoundError("Planilha de população não encontrada em raw_data/Brazil")
    path=candidates[0]; print(f"[INFO] População: {path.name}")
    df=pl.read_excel(path,engine="xlsx2csv").to_pandas()
    code_cols=[c for c in df.columns if "COD" in str(c).upper()]
    pop_cols=[c for c in df.columns if "POP" in str(c).upper() and "TOTAL" in str(c).upper()]
    if not code_cols or not pop_cols: raise ValueError(f"Colunas de código/população não identificadas: {df.columns.tolist()}")
    # Preferir código IBGE completo, ou concatenar UF e município.
    if len(code_cols)>=2:
        uf=next((c for c in code_cols if "UF" in str(c).upper()),None); mun=next((c for c in code_cols if "MUNIC" in str(c).upper()),None)
    else: uf=mun=None
    if uf and mun:
        ids=(df[uf].astype(str).str.extract(r"(\d+)")[0].fillna("").str.zfill(2)+df[mun].astype(str).str.extract(r"(\d+)")[0].fillna("").str.zfill(5))
    else:
        ids=df[code_cols[0]].astype(str).str.extract(r"(\d+)")[0].str.zfill(7)
    pop=pd.to_numeric(df[pop_cols[0]],errors="coerce")
    mapping=dict(zip(ids,pop))
    values=np.array([mapping.get(str(int(c)).zfill(7),np.nan) for c in codes],dtype=float)
    if np.isnan(values).any(): print(f"[AVISO] {np.isnan(values).sum()} municípios sem população; serão ignorados nas taxas.")
    return values

def mean(v):return np.mean(np.stack([np.asarray(x) for x in v]),axis=0)
def persistence(root,codes,n_snap):
    import polars as pl
    f=root/"raw_data/Brazil/cases-brazil-cities-time_changesOnly.csv.gz"; ds=pd.to_datetime(pl.read_csv(f,columns=["date"])["date"].unique().sort().to_numpy()); x=pl.read_csv(f,columns=["date","ibgeID","newCases"])
    dt={pd.Timestamp(v):i for i,v in enumerate(ds)};ci={int(v):i for i,v in enumerate(codes)};a=np.zeros((len(ds),len(codes)),np.float32)
    for r in x.iter_rows(named=True):
        d=pd.Timestamp(r["date"]);c=int(r["ibgeID"])
        if d in dt and c in ci:a[dt[d],ci[c]]=max(float(r["newCases"] or 0),0)
    cut=int((len(ds)-LAG-HORIZON+1)*TRAIN_RATIO);p=np.zeros((n_snap,len(codes),HORIZON),np.float32)
    for s in range(n_snap):p[s]=a[cut+s+LAG-1,:,None]
    return p

def metrics_for(model,real,pred,windows,regions,pop):
    rows=[]; eps=1.0
    for _,w in windows.iterrows():
        s=int(w.snapshot);err=pred[s]-real[s];obs=real[s]
        for rg in REGION_ORDER:
            mask=np.ones(len(regions),bool) if rg=="Brasil" else regions==rg
            valid=mask & np.isfinite(pop) & (pop>0)
            if not valid.any():continue
            # Erro agregado para o total regional e taxa por 100 mil habitantes.
            e=err[valid].sum(axis=0); y=obs[valid].sum(axis=0);total_pop=pop[valid].sum()
            abs_e=np.abs(e)
            rows.append({"model":model,"region":rg,"phase":w.phase,"event":int(w.event),"snapshot":s,
                         "mae_absolute":float(abs_e.mean()),"rmse_absolute":float(np.sqrt(np.mean(e**2))),
                         "mae_per_100k":float(abs_e.mean()/total_pop*100000),"rmse_per_100k":float(np.sqrt(np.mean(e**2))/total_pop*100000),
                         "mean_observed_incidence_per_100k":float(y.mean()/total_pop*100000),
                         "mean_relative_error_percent":float((abs_e/(np.abs(y)+eps)*100).mean()),
                         "median_relative_error_percent":float(np.median(abs_e/(np.abs(y)+eps)*100)),
                         "population":float(total_pop)})
    return rows

def plot(summary,out):
    # Taxa de erro relativo nacional por fase.
    nat=summary[summary.region=="Brasil"]
    fig,ax=plt.subplots(figsize=(11,6));sns.barplot(data=nat,x="phase",y="mean_relative_error_percent",hue="model",order=PHASE_ORDER,ax=ax);ax.set_xlabel("Fase");ax.set_ylabel("Erro relativo médio (\%)");ax.set_title("Erro relativo nacional por fase");fig.tight_layout();fig.savefig(out/"figures/national_relative_error_by_phase.png",dpi=300);plt.close(fig)
    # RMSE por 100 mil habitantes, dividido em dois painéis para leitura.
    regs=["Norte","Nordeste","Sudeste","Sul","Centro-Oeste"]
    for group,name in [(regs[:2],"regional_rmse_per100k_norte_nordeste.png"),(regs[2:],"regional_rmse_per100k_sudeste_sul_co.png")]:
        sub=summary[summary.region.isin(group)];fig,axes=plt.subplots(1,len(group),figsize=(8*len(group),6),squeeze=False);axes=axes[0]
        for ax,rg in zip(axes,group):
            sns.barplot(data=sub[sub.region==rg],x="phase",y="rmse_per_100k",hue="model",order=PHASE_ORDER,ax=ax);ax.set_title(rg);ax.set_xlabel("Fase");ax.set_ylabel("RMSE por 100 mil habitantes")
        fig.tight_layout();fig.savefig(out/f"figures/{name}",dpi=300,bbox_inches="tight");plt.close(fig)
    # Incidência observada contextualiza a escala da epidemia.
    fig,ax=plt.subplots(figsize=(12,6));inc=summary[summary.model=="GCRN"];sns.barplot(data=inc,x="region",y="mean_observed_incidence_per_100k",hue="phase",order=REGION_ORDER,ax=ax);ax.set_xlabel("Região");ax.set_ylabel("Incidência média observada por 100 mil habitantes");ax.tick_params(axis="x",rotation=25);ax.set_title("Incidência média observada nas janelas de eventos");fig.tight_layout();fig.savefig(out/"figures/observed_incidence_per100k_by_region.png",dpi=300);plt.close(fig)

def main():
    r=root();o=output(r);windows_path=o/"tables/national_event_windows.csv"
    if not windows_path.exists():raise FileNotFoundError(f"Execute a análise nacional antes: {windows_path}")
    windows=pd.read_csv(windows_path);data=load_arrays(r);codes=graph_codes(r);regions=np.array([region(c) for c in codes]);pop=load_population(r,codes);n=data["GCRN"]["real"][0].shape[0];pers=persistence(r,codes,n)
    rows=[]
    for m in MODELS:rows+=metrics_for(m,mean(data[m]["real"]),mean(data[m]["pred"]),windows,regions,pop)
    rows+=metrics_for("Persistence",mean(data["GCRN"]["real"]),pers,windows,regions,pop)
    detail=pd.DataFrame(rows);summary=detail.groupby(["model","region","phase"],as_index=False).agg(mae_absolute=("mae_absolute","mean"),rmse_absolute=("rmse_absolute","mean"),mae_per_100k=("mae_per_100k","mean"),rmse_per_100k=("rmse_per_100k","mean"),mean_observed_incidence_per_100k=("mean_observed_incidence_per_100k","mean"),mean_relative_error_percent=("mean_relative_error_percent","mean"),median_relative_error_percent=("median_relative_error_percent","median"),population=("population","first"))
    detail.to_csv(o/"tables/error_rate_national_regional_by_snapshot.csv",index=False);summary[summary.region=="Brasil"].to_csv(o/"tables/error_rate_national_by_phase.csv",index=False);summary[summary.region!="Brasil"].to_csv(o/"tables/error_rate_regional_by_phase.csv",index=False);plot(summary,o);print(f"[INFO] Taxas de erro salvas em {o/'tables'}")
if __name__=="__main__":main()