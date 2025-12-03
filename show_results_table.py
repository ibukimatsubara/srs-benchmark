import json
import sys

def show_metrics(file_path):
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    
    metrics = [d['metrics'] for d in data]
    sizes = [d['size'] for d in data]
    total_size = sum(sizes)
    
    weighted_metrics = {}
    for key in metrics[0].keys():
        weighted_sum = sum(m[key] * s for m, s in zip(metrics, sizes))
        weighted_metrics[key] = weighted_sum / total_size
    
    return {
        'model': file_path.split('/')[-1].replace('.jsonl', ''),
        'users': len(data),
        'reviews': total_size,
        **weighted_metrics
    }

if __name__ == "__main__":
    files = ["result/FSRS-6.jsonl", "result/FSRS-6-cefr.jsonl"]
    
    results = []
    for file_path in files:
        try:
            results.append(show_metrics(file_path))
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            sys.exit(1)
    
    # Print table
    print("\n" + "="*90)
    print(f"{'Model':<20} {'Users':>8} {'Reviews':>12} {'RMSE':>10} {'LogLoss':>10} {'RMSE(bins)':>12} {'ICI':>10}")
    print("="*90)
    
    for r in results:
        print(f"{r['model']:<20} {r['users']:>8,} {r['reviews']:>12,} {r['RMSE']:>10.6f} {r['LogLoss']:>10.6f} {r['RMSE(bins)']:>12.6f} {r['ICI']:>10.6f}")
    
    # Calculate differences
    if len(results) == 2:
        print("="*90)
        print(f"{'Difference (CEFR - Base)':<20}")
        base = results[0]
        cefr = results[1]
        
        rmse_diff = cefr['RMSE'] - base['RMSE']
        logloss_diff = cefr['LogLoss'] - base['LogLoss']
        rmse_bins_diff = cefr['RMSE(bins)'] - base['RMSE(bins)']
        ici_diff = cefr['ICI'] - base['ICI']
        
        print(f"{'':20} {'':>8} {'':>12} {rmse_diff:>+10.6f} {logloss_diff:>+10.6f} {rmse_bins_diff:>+12.6f} {ici_diff:>+10.6f}")
        print("="*90)
        
        print(f"\n{'':20} {'':>8} {'':>12} {rmse_diff/base['RMSE']*100:>+9.2f}% {logloss_diff/base['LogLoss']*100:>+9.2f}% {rmse_bins_diff/base['RMSE(bins)']*100:>+11.2f}% {ici_diff/base['ICI']*100:>+9.2f}%")
