"""
Análise COVID-19 Brasil — lags_14_out_14.
Versão com relatório Markdown completo.

Detecta picos nacionais (soma de todos os municípios), classifica janelas
de subida/pico/declínio, calcula métricas por modelo, seleciona as 5 cidades
mais importantes pelas métricas de rede e gera relatório Markdown.
"""
import gc
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
LAG, HORIZON, TRAIN_RATIO = 14, 14, 0.8
N_EVENTS = 4
EVENT_WINDOW = 21
SURGE_WINDOW = 14
TOP_N = 5


def project_root():
    p = Path(__file__).resolve().parent.parent
    data = Path(str(p).replace("/media/work/", "/media/data/", 1))
    return data if (data / "results_daily" / "Brazil").exists() else p


def paths(root):
    out = root / "results_daily/results_analysis/Brazil/covid_lags_14_out_14"
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    return out


def load_predictions(root):
    base = root / "results_daily/Brazil/regression/grafo_original/backbone_alpha_1/k_1_hc_256"
    result = {}
    for model in ["GCRN", "GCLSTM", "LSTM"]:
        d = base / model / "lags_14_out_14"
        if not d.exists():
            raise FileNotFoundError(f"Configuração lags_14_out_14 ausente: {d}")
        real = sorted(d.glob("y_real_no_norm_rep_*.npy"))
        pred = [d / x.name.replace("y_real_no_norm_rep_", "y_pred_no_norm_rep_") for x in real]
        pred = [p for p in pred if p.exists()]
        if not real or len(real) != len(pred):
            raise FileNotFoundError(f"Pares y_real/y_pred incompletos para {model}: {d}")
        result[model] = {"real": [np.load(x, mmap_mode="r") for x in real],
                         "pred": [np.load(x, mmap_mode="r") for x in pred]}
        print(f"[INFO] {model}: {len(real)} repetições, shape={result[model]['real'][0].shape}")
    return result


def load_dates(root, n_snap):
    import polars as pl
    f = root / "raw_data/Brazil/cases-brazil-cities-time_changesOnly.csv.gz"
    dates = pl.read_csv(f, columns=["date"])["date"].unique().sort().to_numpy()
    n_samples = len(dates) - LAG - HORIZON + 1
    test_start = int(n_samples * TRAIN_RATIO)
    idx = test_start + np.arange(n_snap)[:, None] + LAG + np.arange(HORIZON)[None, :]
    idx = np.clip(idx, 0, len(dates) - 1)
    flat_idx = idx.flatten()
    flat_dates = pd.to_datetime(dates[flat_idx]).to_numpy()
    return flat_dates.reshape(idx.shape)


def mean_arrays(arrays):
    return np.mean(np.stack([np.asarray(x) for x in arrays], axis=0), axis=0)


def aggregate_national(y):
    return y.sum(axis=1)


def detect_national_events(y_true, dates):
    total = aggregate_national(y_true)
    daily = total[:, 0]
    smooth = pd.Series(daily).rolling(7, center=True, min_periods=1).mean().to_numpy()
    peaks = []
    for i in range(1, len(smooth)-1):
        if smooth[i] >= smooth[i-1] and smooth[i] >= smooth[i+1]:
            peaks.append(i)
    peaks = sorted(peaks, key=lambda i: smooth[i], reverse=True)
    chosen = []
    min_gap = 35
    for i in peaks:
        if all(abs(i-j) >= min_gap for j in chosen):
            chosen.append(i)
        if len(chosen) == N_EVENTS:
            break
    chosen.sort()
    rows = []
    for rank, i in enumerate(chosen, 1):
        left = max(0, i-SURGE_WINDOW)
        slope = (smooth[i] - smooth[left]) / max(i-left, 1)
        rows.append({"event": rank, "snapshot": i, "date": dates[i, 0],
                     "peak_total_cases": smooth[i], "pre_peak_total": smooth[left],
                     "absolute_rise": smooth[i]-smooth[left], "daily_rise": slope})
    return pd.DataFrame(rows), pd.DataFrame({"snapshot": np.arange(len(daily)),
                                               "date": dates[:, 0], "national_total": daily,
                                               "national_smooth7": smooth})


def event_windows(events, n):
    rows=[]
    for _, e in events.iterrows():
        p=int(e.snapshot)
        for s in range(max(0,p-EVENT_WINDOW), min(n,p+EVENT_WINDOW+1)):
            d=s-p
            phase="surge" if d < -7 else ("peak" if d <= 7 else "decline")
            rows.append({"event":int(e.event), "snapshot":s, "days_from_peak":d, "phase":phase})
    return pd.DataFrame(rows).drop_duplicates(["event","snapshot"])


def persistence(y):
    out=np.empty_like(y)
    out[:,:,0]=y[:,:,0]
    out[:,:,1:]=y[:,:,:-1]
    return out


def event_metrics(data, windows, out):
    rows=[]
    for model, d in data.items():
        real=mean_arrays(d["real"]); pred=mean_arrays(d["pred"])
        for _, w in windows.iterrows():
            s=int(w.snapshot)
            e=(pred[s]-real[s]).sum(axis=0)
            rows.append({"model":model,"event":int(w.event),"phase":w.phase,
                         "snapshot":s,"date":w.get("date", ""),
                         "RMSE_national":float(np.sqrt(np.mean(e**2))),
                         "MAE_national":float(np.mean(np.abs(e)))})
        p=persistence(real)
        for _, w in windows.iterrows():
            s=int(w.snapshot); e=(p[s]-real[s]).sum(axis=0)
            rows.append({"model":"Persistence","event":int(w.event),"phase":w.phase,
                         "snapshot":s,"date":"","RMSE_national":float(np.sqrt(np.mean(e**2))),
                         "MAE_national":float(np.mean(np.abs(e)))})
    df=pd.DataFrame(rows)
    summary=df.groupby(["model","phase"])[["RMSE_national","MAE_national"]].mean().reset_index()
    df.to_csv(out/"tables/event_metrics_by_snapshot.csv",index=False)
    summary.to_csv(out/"tables/event_metrics_summary.csv",index=False)
    return summary


def load_network(root):
    d=root/"raw_data/Brazil/in"; frames=[]
    for metric in ["betweenness","strength","degree","closeness"]:
        f=d/f"{metric}.csv"
        if f.exists():
            x=pd.read_csv(f,sep=";",header=None,names=["city","ibgeID","value"])
            x["ibgeID"]=pd.to_numeric(x.ibgeID,errors="coerce").astype("Int64")
            x=x.rename(columns={"value":metric})[["ibgeID",metric,"city"]]
            frames.append(x)
    if not frames: raise FileNotFoundError(f"Métricas não encontradas em {d}")
    result=frames[0]
    for x in frames[1:]: result=result.merge(x.drop(columns="city"),on="ibgeID",how="outer")
    return result


def select_top(network,out):
    metric_cols=[c for c in ["betweenness","strength","degree","closeness"] if c in network]
    long=[]
    for m in metric_cols:
        x=network.nlargest(TOP_N,m)[["ibgeID","city",m]].rename(columns={m:"value"})
        x["metric"]=m; long.append(x)
    alltop=pd.concat(long,ignore_index=True)
    selected=(alltop.groupby(["ibgeID","city"]).agg(n_metrics_in_top=("metric","count"),mean_value=("value","mean"))
              .reset_index().sort_values(["n_metrics_in_top","mean_value"],ascending=False))
    selected["rank"]=np.arange(1,len(selected)+1)
    selected.head(TOP_N).to_csv(out/"tables/top5_important_cities.csv",index=False)
    alltop.to_csv(out/"tables/top5_by_network_metric.csv",index=False)
    return selected.head(TOP_N)


def plot_national_series(national, events, out):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(national["date"], national["national_total"], label="Total diário", color="lightgray", lw=1)
    ax.plot(national["date"], national["national_smooth7"], label="Média móvel 7 dias", color="black", lw=2)
    for _, e in events.iterrows():
        ax.axvline(e["date"], color="red", ls="--", alpha=0.6)
        ax.annotate(f"Evento {int(e['event'])}", xy=(e["date"], e["peak_total_cases"]),
                    xytext=(5, 10), textcoords="offset points", fontsize=9)
    ax.set_title("Série nacional de casos e eventos detectados", weight="bold")
    ax.set_ylabel("Casos (soma nacional)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out/"figures/national_series_events.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_event_metrics(summary, out):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, metric in zip(axes, ["RMSE_national", "MAE_national"]):
        pivot = summary.pivot(index="phase", columns="model", values=metric)
        order = ["surge", "peak", "decline"]
        pivot = pivot.reindex(order)
        pivot.plot(kind="bar", ax=ax, width=0.8)
        ax.set_title(metric.replace("_", " "), weight="bold")
        ax.set_xlabel("Fase")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=0)
        ax.legend(title="Modelo", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.suptitle("Desempenho por fase do evento nacional", weight="bold")
    plt.tight_layout()
    plt.savefig(out/"figures/event_metrics_by_phase.png", dpi=300, bbox_inches="tight")
    plt.close()


def report(out, events, summary, important, national):
    text = "# Análise COVID-19 — Brasil — lags_14_out_14\n\n"
    text += "## Objetivo\n\n"
    text += "Avaliar previsões no horizonte de 14 dias em torno das principais subidas e picos nacionais, definidos pela soma dos casos de todos os municípios.\n\n"
    
    text += "## Eventos nacionais detectados\n\n"
    text += "| Evento | Data | Pico suavizado | Alta em 14 dias | Aumento diário |\n"
    text += "|---|---|---:|---:|---:|\n"
    for _, r in events.iterrows():
        text += f"| {int(r.event)} | {r.date} | {r.peak_total_cases:.2f} | {r.absolute_rise:.2f} | {r.daily_rise:.2f} |\n"
    
    text += "\n## Desempenho por fase do evento\n\n"
    text += "| Modelo | Fase | RMSE nacional médio | MAE nacional médio |\n"
    text += "|---|---|---:|---:|\n"
    for _, r in summary.iterrows():
        text += f"| {r.model} | {r.phase} | {r.RMSE_national:.2f} | {r.MAE_national:.2f} |\n"
    
    text += "\n## Cinco cidades importantes\n\n"
    text += "| Rank | Cidade | IBGE | Métricas no top 5 | Valor médio |\n"
    text += "|---:|---|---:|---:|---:|\n"
    for _, r in important.iterrows():
        text += f"| {int(r['rank'])} | {r.city} | {int(r.ibgeID)} | {int(r.n_metrics_in_top)} | {r.mean_value:.2f} |\n"
    
    text += "\n## Interpretação\n\n"
    text += "Os eventos são nacionais: cada pico corresponde a uma elevação da soma de casos, e não ao pico isolado de um município. A fase `surge` representa os snapshots anteriores ao pico; `peak`, a janela central; e `decline`, os snapshots posteriores.\n\n"
    text += "A métrica nacional é calculada somando os erros municipais, medindo o erro da previsão do total nacional. Para avaliar o desempenho em cidades específicas, consulte as tabelas de cidades importantes.\n\n"
    
    text += "## Figuras\n\n"
    text += "![Série nacional e eventos](figures/national_series_events.png)\n\n"
    text += "![Desempenho por fase](figures/event_metrics_by_phase.png)\n\n"
    
    text += "## Arquivos gerados\n\n"
    text += "- `tables/national_events.csv`\n"
    text += "- `tables/national_series.csv`\n"
    text += "- `tables/national_event_windows.csv`\n"
    text += "- `tables/event_metrics_by_snapshot.csv`\n"
    text += "- `tables/event_metrics_summary.csv`\n"
    text += "- `tables/top5_important_cities.csv`\n"
    text += "- `tables/top5_by_network_metric.csv`\n"
    text += "- `figures/national_series_events.png`\n"
    text += "- `figures/event_metrics_by_phase.png`\n"
    
    (out / "report_national_events.md").write_text(text, encoding="utf-8")
    print(f"[INFO] Relatório salvo em: {out / 'report_national_events.md'}")


def main():
    root = project_root(); out = paths(root)
    print(f"[INFO] Projeto: {root}\n[INFO] Configuração fixa: lags_{LAG}_out_{HORIZON}")
    data = load_predictions(root)
    n = data["GCRN"]["real"][0].shape[0]
    dates = load_dates(root, n)
    real = mean_arrays(data["GCRN"]["real"])
    events, national = detect_national_events(real, dates)
    events.to_csv(out/"tables/national_events.csv", index=False)
    national.to_csv(out/"tables/national_series.csv", index=False)
    windows = event_windows(events, n)
    windows = windows.merge(national[["snapshot", "date"]], on="snapshot", how="left")
    windows.to_csv(out/"tables/national_event_windows.csv", index=False)
    summary = event_metrics(data, windows, out)
    network = load_network(root); important = select_top(network, out)
    plot_national_series(national, events, out)
    plot_event_metrics(summary, out)
    report(out, events, summary, important, national)
    print(f"[INFO] Eventos nacionais detectados: {len(events)}")
    print(f"[INFO] Resultados salvos em: {out}")

if __name__ == "__main__":
    main()