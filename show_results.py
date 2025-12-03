import json
import sys

def show_metrics(file_path):
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    
    # Extract metrics
    metrics = [d['metrics'] for d in data]
    sizes = [d['size'] for d in data]
    
    # Calculate weighted average
    total_size = sum(sizes)
    weighted_metrics = {}
    
    for key in metrics[0].keys():
        weighted_sum = sum(m[key] * s for m, s in zip(metrics, sizes))
        weighted_metrics[key] = weighted_sum / total_size
    
    print(f"\n{'='*60}")
    print(f"Model: {file_path.split('/')[-1].replace('.jsonl', '')}")
    print(f"{'='*60}")
    print(f"Users: {len(data)}")
    print(f"Total reviews: {total_size:,}")
    print(f"\nWeighted Average Metrics:")
    for key, value in weighted_metrics.items():
        print(f"  {key:15s}: {value:.6f}")

if __name__ == "__main__":
    files = sys.argv[1:] if len(sys.argv) > 1 else [
        "result/FSRS-6.jsonl",
        "result/FSRS-6-cefr.jsonl"
    ]
    
    for file_path in files:
        try:
            show_metrics(file_path)
        except FileNotFoundError:
            print(f"\nFile not found: {file_path}")
        except Exception as e:
            print(f"\nError processing {file_path}: {e}")
