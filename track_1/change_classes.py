import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('submission_246->200_aglomerative.csv')

fig, ax = plt.subplots(figsize=(12, 8))

def update_plot():
    ax.clear()  
    scatter = ax.scatter(df['TSNE 1'], df['TSNE 2'], c=df['prediction'], cmap='Spectral', alpha=0.6)

    # Добавляем подписи кластеров
    for i, (x, y) in df[['TSNE 1', 'TSNE 2']].iterrows():
        ax.text(x, y, str(df['prediction'][i]), fontsize=8, color='black', alpha=0.7)

    cluster_counts = df['prediction'].value_counts()
    for cluster_id, count in cluster_counts.items():
        cluster_points = df[df['prediction'] == cluster_id]
        mean_x = cluster_points['TSNE 1'].mean()
        mean_y = cluster_points['TSNE 2'].mean()
        
        ax.text(mean_x, mean_y, str(count), fontsize=12, color='red', weight='bold', ha='center', va='center')

    ax.set_title('Adjusted Hierarchical Clustering Results with Object Counts')
    ax.set_xlabel('TSNE 1')
    ax.set_ylabel('TSNE 2')
    ax.grid()
    plt.draw() 

def onclick(event):
    if event.xdata is None or event.ydata is None:
        return

    distances = np.sqrt((df['TSNE 1'] - event.xdata) ** 2 + (df['TSNE 2'] - event.ydata) ** 2)
    idx = distances.idxmin()

    print(f'Current class {idx}: {df.loc[idx, "prediction"]}')

    try:
        new_class = int(input('New class: '))
    except ValueError:
        print("Write the number")
        return

    df.at[idx, 'prediction'] = new_class
    df.to_csv('submission_246->200_aglomerative.csv', index=False)
    update_plot()

update_plot()

cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()
