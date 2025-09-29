import pandas as pd
df = pd.read_parquet('data/features.parquet')

# Check reorder rates by different segments
print('Reorder rate by times_bought:')
reorder_by_bought = df.groupby('times_bought')['y'].agg(['count', 'mean']).head(10)
print(reorder_by_bought)
print()

print('Products that were bought multiple times (likely to be reordered):')
multi_bought = df[df['times_bought'] > 1]
print(f'Multi-bought products: {len(multi_bought)} ({len(multi_bought)/len(df)*100:.1f}%)')
print(f'Reorder rate for multi-bought: {multi_bought["y"].mean():.3f}')
print()

print('Products bought only once (cannot be reordered):')
single_bought = df[df['times_bought'] == 1]
print(f'Single-bought products: {len(single_bought)} ({len(single_bought)/len(df)*100:.1f}%)')
print(f'Reorder rate for single-bought: {single_bought["y"].mean():.3f}')