#!/bin/bash

# Results analyzer for KVBM performance testing with long_doc_qa benchmark
# Usage: ./analyze_results.sh [timestamp] [--folder <path>] [--analyze-all] [--include-configs <list>] [--baseline-config <name>] [--show-improvements]
#   timestamp:    Specific timestamp to analyze (optional)
#   --folder:     Directory containing result files (optional, defaults to ./results)
#   --analyze-all: Analyze all *_summary.csv files in directory regardless of naming pattern
#   --include-configs: Comma-separated list of configs to include (e.g., baseline,production,lmcache)
#   --baseline-config: Configuration to use as reference for comparisons (defaults to 'baseline')
#   --show-improvements: Enable improvement markers and connecting lines on charts (default: off)
#
# Examples:
#   ./analyze_results.sh                                    # Auto-find latest in ./results
#   ./analyze_results.sh 20250905_180225                   # Specific timestamp in ./results
#   ./analyze_results.sh --folder /path/to/results         # Auto-find latest in custom folder
#   ./analyze_results.sh 20250905_180225 --folder /custom  # Specific timestamp in custom folder
#   ./analyze_results.sh --analyze-all                     # Analyze all summary files in ./results
#   ./analyze_results.sh --analyze-all --folder /custom    # Analyze all summary files in custom folder
#   ./analyze_results.sh --analyze-all --include-configs production,lmcache  # Only analyze production and lmcache configs
#   ./analyze_results.sh --analyze-all --baseline-config production  # Use production as reference for comparisons
#   ./analyze_results.sh --analyze-all --show-improvements  # Charts with improvement annotations

set -e

# Default values
RESULTS_DIR="./results"
SPECIFIED_TIMESTAMP=""
ANALYZE_ALL=false
INCLUDE_CONFIGS=""
BASELINE_CONFIG="baseline"
SHOW_IMPROVEMENTS=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --folder)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --analyze-all)
            ANALYZE_ALL=true
            shift
            ;;
        --include-configs)
            INCLUDE_CONFIGS="$2"
            shift 2
            ;;
        --baseline-config)
            BASELINE_CONFIG="$2"
            shift 2
            ;;
        --show-improvements)
            SHOW_IMPROVEMENTS=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [timestamp] [--folder <path>] [--analyze-all] [--include-configs <list>] [--baseline-config <name>] [--show-improvements]"
            echo ""
            echo "Arguments:"
            echo "  timestamp         Specific timestamp to analyze (optional)"
            echo "  --folder          Directory containing result files (optional, defaults to ./results)"
            echo "  --analyze-all     Analyze all *_summary.csv files regardless of naming pattern"
            echo "  --include-configs Comma-separated list of configs to include (e.g., baseline,production,lmcache)"
            echo "  --baseline-config Configuration to use as reference for comparisons (defaults to 'baseline')"
            echo "  --show-improvements Enable improvement markers and connecting lines on charts (default: off)"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Auto-find latest in ./results"
            echo "  $0 20250905_180225                   # Specific timestamp in ./results"
            echo "  $0 --folder /path/to/results         # Auto-find latest in custom folder"
            echo "  $0 20250905_180225 --folder /custom  # Specific timestamp in custom folder"
            echo "  $0 --analyze-all                     # Analyze all summary files in ./results"
            echo "  $0 --analyze-all --folder /custom    # Analyze all summary files in custom folder"
            echo "  $0 --analyze-all --include-configs production,lmcache  # Only analyze production and lmcache configs"
            echo "  $0 --analyze-all --baseline-config production  # Use production as reference for comparisons"
            echo "  $0 --analyze-all --show-improvements  # Charts with improvement annotations"
            exit 0
            ;;
        *)
            # If it looks like a timestamp (digits and underscores), treat it as such
            if [[ "$1" =~ ^[0-9_]+$ ]]; then
                SPECIFIED_TIMESTAMP="$1"
            else
                echo "❌ Unknown argument: $1"
                echo "Use --help for usage information"
                exit 1
            fi
            shift
            ;;
    esac
done

# Validate results directory
if [[ ! -d "$RESULTS_DIR" ]]; then
    echo "❌ Results directory does not exist: $RESULTS_DIR"
    exit 1
fi

# Create Python analysis script
cat > /tmp/analyze_long_doc_qa.py << 'EOF'
import csv
import sys
import os
import statistics
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

def read_summary_data(results_dir: str, test_prefix: str) -> Dict[int, Dict[str, float]]:
    """Read data from summary CSV file."""
    summary_file = f"{results_dir}/{test_prefix}_summary.csv"
    
    if not os.path.exists(summary_file):
        print(f"❌ Summary file not found: {summary_file}")
        return {}
    
    data = {}
    
    try:
        with open(summary_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    isl = int(row['ISL'])
                    mean_ttft = float(row['Mean_TTFT'])
                    query_time = float(row['Query_Time'])
                    prompt_count = int(row['Prompt_Count'])
                    
                    data[isl] = {
                        'mean_ttft': mean_ttft,
                        'query_time': query_time,
                        'prompt_count': prompt_count
                    }
                    
                    print(f"ISL {isl:6d}: Mean TTFT = {mean_ttft:.3f}s, Query Time = {query_time:.3f}s, Prompts = {prompt_count}")
                except (ValueError, KeyError) as e:
                    print(f"Error parsing row: {e}")
                    continue
    except Exception as e:
        print(f"Error reading summary file: {e}")
        return {}
    
    return data

def analyze_single(results_dir: str, test_prefix: str):
    """Analyze a single test."""
    print(f"=== Analysis for {test_prefix} ===")
    data = read_summary_data(results_dir, test_prefix)
    
    if not data:
        print("No data found!")
        return
    
    print(f"\nSummary:")
    print(f"ISL values tested: {len(data)}")
    
    ttft_values = [v['mean_ttft'] for v in data.values()]
    if ttft_values:
        print(f"Overall mean TTFT: {statistics.mean(ttft_values):.3f}s")
        print(f"Min TTFT: {min(ttft_values):.3f}s")
        print(f"Max TTFT: {max(ttft_values):.3f}s")

def create_comparison_chart(results_dir: str, config_data: dict, configs: list, 
                           metric: str, ylabel: str, title_suffix: str, 
                           baseline_config: str = 'baseline', show_improvements: bool = True):
    """Create a comparison chart for a specific metric."""
    
    # Predefined styles for common configuration names
    predefined_styles = {
        'baseline': {'color': '#d62728', 'marker': 'o', 'label': 'vLLM (No Cache)'},
        'production': {'color': '#2ca02c', 'marker': 's', 'label': 'vLLM + KVBM Connector'},
        'connector': {'color': '#ff7f0e', 'marker': '^', 'label': 'Connector Only'},
        'lmcache': {'color': '#1f77b4', 'marker': 'D', 'label': 'LMCache'},
        'vllm_prefix_cache': {'color': '#9467bd', 'marker': 'v', 'label': 'vLLM Prefix Cache'}
    }
    
    # Extended color palette for additional configs
    color_palette = [
        {'color': '#8c564b', 'marker': '<'},  # Brown
        {'color': '#e377c2', 'marker': '>'},  # Pink
        {'color': '#7f7f7f', 'marker': 'p'},  # Gray
        {'color': '#bcbd22', 'marker': '*'},  # Olive
        {'color': '#17becf', 'marker': 'h'},  # Cyan
    ]
    
    # Assign styles to configs
    config_styles = {}
    color_index = 0
    for i, config in enumerate(configs):
        if config in predefined_styles:
            config_styles[config] = predefined_styles[config]
        else:
            palette_idx = color_index % len(color_palette)
            config_styles[config] = {
                'color': color_palette[palette_idx]['color'],
                'marker': color_palette[palette_idx]['marker'],
                'label': config.capitalize()
            }
            color_index += 1
    
    plt.figure(figsize=(12, 7))
    
    # Plot each configuration
    for config in configs:
        data = config_data[config]
        style = config_styles[config]
        
        isl_list = sorted(data.keys())
        values = [data[isl][metric] for isl in isl_list]
        
        # Convert ISL to K for x-axis labels
        isl_k = [isl / 1000 for isl in isl_list]
        
        plt.plot(isl_k, values, f"{style['marker']}-",
                linewidth=3, markersize=8,
                label=style['label'], color=style['color'], alpha=0.9)
        
        # Add value annotations
        for x, y in zip(isl_k, values):
            plt.annotate(f'{y:.3f}', (x, y),
                        textcoords="offset points",
                        xytext=(0, 10), ha='center', fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor=style['color'], alpha=0.3))
    
    # Add improvement annotations if enabled
    if show_improvements and baseline_config in configs and baseline_config in config_data:
        baseline_data = config_data[baseline_config]
        
        for config in configs:
            if config == baseline_config or config not in config_data:
                continue
            
            other_data = config_data[config]
            style = config_styles[config]
            
            # Find common ISL values
            common_isl = sorted(set(baseline_data.keys()) & set(other_data.keys()))
            
            for isl in common_isl:
                baseline_val = baseline_data[isl][metric]
                other_val = other_data[isl][metric]
                
                # Calculate improvement (positive = better performance)
                improvement = ((baseline_val - other_val) / baseline_val) * 100
                
                if abs(improvement) > 1:  # Only show if improvement is > 1%
                    isl_k = isl / 1000
                    
                    # Calculate speedup
                    speedup = baseline_val / other_val if other_val > 0 else 1.0
                    
                    # Determine arrow direction
                    if improvement > 0:
                        arrow_symbol = '↓'
                        if speedup >= 2.0:
                            improvement_text = f'{arrow_symbol}{improvement:.0f}% ({speedup:.1f}X)'
                        else:
                            improvement_text = f'{arrow_symbol}{improvement:.0f}%'
                    else:
                        arrow_symbol = '↑'
                        slowdown = other_val / baseline_val
                        if slowdown >= 2.0:
                            improvement_text = f'{arrow_symbol}{abs(improvement):.0f}% ({slowdown:.1f}X slower)'
                        else:
                            improvement_text = f'{arrow_symbol}{abs(improvement):.0f}%'
                    
                    # Draw connecting line
                    plt.plot([isl_k, isl_k], [baseline_val, other_val],
                            color=style['color'], linewidth=1, alpha=0.5, linestyle='--')
                    
                    # Add improvement annotation above the chart
                    y_min, y_max = plt.ylim()
                    y_range = y_max - y_min
                    annotation_y = y_max + (y_range * 0.1)
                    
                    plt.annotate(improvement_text, (isl_k, annotation_y),
                                ha='center', va='bottom', fontsize=9, fontweight='bold',
                                color=style['color'],
                                bbox=dict(boxstyle="round,pad=0.3", facecolor='white',
                                        edgecolor=style['color'], alpha=0.9))
    
    # Customize plot
    plt.xlabel('Input Sequence Length (K tokens)', fontsize=12, fontweight='bold')
    plt.ylabel(ylabel, fontsize=12, fontweight='bold')
    plt.title(f'KVBM Performance: {title_suffix}', fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=10, loc='best', frameon=True, fancybox=True, shadow=True)
    
    # Set x-axis to show ISL values in K
    all_isl = set()
    for config in configs:
        all_isl.update(config_data[config].keys())
    
    if all_isl:
        isl_sorted = sorted(all_isl)
        isl_k_labels = [isl / 1000 for isl in isl_sorted]
        plt.xticks(isl_k_labels, labels=[f'{int(k)}K' for k in isl_k_labels])
    
    plt.tight_layout()
    
    # Save chart
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    chart_name = f"kvbm_{metric}_{timestamp}.png"
    chart_path = f"{results_dir}/{chart_name}"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"📊 Chart saved: {chart_path}")
    
    return chart_path

def create_charts(results_dir: str, config_data: dict, configs: list, 
                 baseline_config: str = 'baseline', show_improvements: bool = True):
    """Create comparison charts for all metrics."""
    
    print(f"\n📊 Creating comparison charts...")
    chart_paths = []
    
    # Chart 1: Mean TTFT
    chart1 = create_comparison_chart(results_dir, config_data, configs,
                                     'mean_ttft', 'Mean TTFT (seconds)',
                                     'Mean Time To First Token vs ISL',
                                     baseline_config, show_improvements)
    chart_paths.append(chart1)
    
    # Chart 2: Query Time
    chart2 = create_comparison_chart(results_dir, config_data, configs,
                                     'query_time', 'Query Time (seconds)',
                                     'Total Query Time vs ISL',
                                     baseline_config, show_improvements)
    chart_paths.append(chart2)
    
    print(f"✅ Generated {len(chart_paths)} charts")
    return chart_paths

def print_comparison_table(config_data: dict, configs: list):
    """Print a comparison table for all configurations."""
    
    total_width = 120
    
    print(f"\n{'='*total_width}")
    title = "KVBM Performance Comparison Summary"
    title_padding = (total_width - len(title)) // 2
    print(f"{' '*title_padding}{title}")
    print(f"{'='*total_width}")
    
    # Get all ISL values
    all_isl = set()
    for data in config_data.values():
        all_isl.update(data.keys())
    all_isl = sorted(all_isl)
    
    if not all_isl:
        print("No data found!")
        return
    
    # Define column widths
    isl_width = 8
    value_width = 15
    percent_width = 13
    
    # Print header
    header = f"{'ISL':<{isl_width}}"
    for config in configs:
        header += f" {config.capitalize() + ' TTFT':<{value_width}}"
    
    if 'baseline' in configs and len(configs) > 1:
        for config in configs:
            if config != 'baseline':
                header += f" {config[:4].upper() + ' Improve':<{percent_width}}"
    
    print(header)
    print("-" * len(header))
    
    # Print data rows
    for isl in all_isl:
        row = f"{isl:<{isl_width}}"
        values = {}
        
        # Get values for each config
        for config in configs:
            if isl in config_data[config]:
                ttft = config_data[config][isl]['mean_ttft']
                values[config] = ttft
                row += f" {ttft:<{value_width}.3f}"
            else:
                row += f" {'N/A':<{value_width}}"
        
        # Calculate improvements
        if 'baseline' in values:
            for config in configs:
                if config != 'baseline' and config in values:
                    improvement = ((values['baseline'] - values[config]) / values['baseline']) * 100
                    row += f" {improvement:<{percent_width-2}.1f} %"
        
        print(row)
    
    # Print summary statistics
    print("\n" + "="*total_width)
    print("Summary Statistics")
    print("="*total_width)
    
    for config in configs:
        ttft_values = [data['mean_ttft'] for data in config_data[config].values()]
        query_times = [data['query_time'] for data in config_data[config].values()]
        
        print(f"\n{config.capitalize()}:")
        print(f"  Mean TTFT: {statistics.mean(ttft_values):.3f}s (min: {min(ttft_values):.3f}s, max: {max(ttft_values):.3f}s)")
        print(f"  Mean Query Time: {statistics.mean(query_times):.3f}s")

def main():
    if len(sys.argv) < 3:
        print("Usage: python analyze_long_doc_qa.py <results_dir> <config_prefixes> <configs> [baseline_config] [show_improvements]")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    config_prefixes = sys.argv[2].split(',')
    configs = sys.argv[3].split(',')
    baseline_config = sys.argv[4] if len(sys.argv) > 4 else 'baseline'
    show_improvements = len(sys.argv) > 5 and sys.argv[5].lower() == 'true'
    
    print(f"Analyzing configurations: {configs}")
    print(f"Using prefixes: {config_prefixes}")
    
    # Read data for each configuration
    config_data = {}
    for config, prefix in zip(configs, config_prefixes):
        print(f"\n📊 Reading data for {config} ({prefix}):")
        data = read_summary_data(results_dir, prefix)
        if data:
            config_data[config] = data
        else:
            print(f"⚠️  No data found for {config}")
    
    if not config_data:
        print("❌ No data found for any configuration")
        sys.exit(1)
    
    # Create charts
    create_charts(results_dir, config_data, list(config_data.keys()), baseline_config, show_improvements)
    
    # Print comparison table
    print_comparison_table(config_data, list(config_data.keys()))

if __name__ == "__main__":
    main()
EOF

if [[ "$ANALYZE_ALL" == "true" ]]; then
    echo "🔍 Analyzing ALL summary files in directory..."
    echo "📁 Looking in directory: $RESULTS_DIR"
elif [[ -n "$SPECIFIED_TIMESTAMP" ]]; then
    echo "🔍 Analyzing results for specified timestamp: $SPECIFIED_TIMESTAMP"
    echo "📁 Looking in directory: $RESULTS_DIR"
else
    echo "🔍 Finding latest test results for each configuration..."
    echo "📁 Looking in directory: $RESULTS_DIR"
fi

# Find latest test file for each configuration or use specified timestamp
find_latest_config() {
    local config=$1
    if [[ -n "$SPECIFIED_TIMESTAMP" ]]; then
        # Check if files exist for the specified timestamp
        if find "$RESULTS_DIR" -name "${config}_*${SPECIFIED_TIMESTAMP}_summary.csv" 2>/dev/null | head -1 | grep -q .; then
            echo "$SPECIFIED_TIMESTAMP"
        else
            echo ""
        fi
    else
        # Find latest
        find "$RESULTS_DIR" -name "${config}_*_summary.csv" 2>/dev/null | \
        sed "s|.*/${config}_\(.*\)_summary.csv|\1|" | sed 's/8B_//' | sed 's/70B_//' | sort -u | tail -1
    fi
}

# Also check for files with different prefixes
find_config_with_prefix() {
    local config=$1
    if [[ -n "$SPECIFIED_TIMESTAMP" ]]; then
        local found_file=$(find "$RESULTS_DIR" -name "*${config}*${SPECIFIED_TIMESTAMP}_summary.csv" 2>/dev/null | head -1)
        if [[ -n "$found_file" ]]; then
            basename "$found_file" | sed 's/_summary.csv$//'
        else
            echo ""
        fi
    else
        find "$RESULTS_DIR" -name "*${config}*_summary.csv" 2>/dev/null | \
        sed "s|.*/\(.*\)_summary.csv|\1|" | sort -u | tail -1
    fi
}

# Function to analyze all summary files in directory
analyze_all_files() {
    echo ""
    echo "🔍 Discovering all summary files in directory..."
    
    # Find all *_summary.csv files
    local all_files=($(find "$RESULTS_DIR" -name "*_summary.csv" 2>/dev/null | sort))
    
    if [[ ${#all_files[@]} -eq 0 ]]; then
        echo "❌ No *_summary.csv files found in $RESULTS_DIR"
        exit 1
    fi
    
    echo "Found ${#all_files[@]} summary files"
    
    # Extract unique prefixes from all files
    local prefixes=()
    local configs=()
    
    for file in "${all_files[@]}"; do
        local basename_file=$(basename "$file")
        local prefix=$(echo "$basename_file" | sed 's/_summary.csv$//')
        
        # Check if this prefix is already in our list
        local found=false
        for existing_prefix in "${prefixes[@]}"; do
            if [[ "$existing_prefix" == "$prefix" ]]; then
                found=true
                break
            fi
        done
        
        if [[ "$found" == "false" ]]; then
            prefixes+=("$prefix")
            
            # Extract config name from prefix
            local config_name=$(echo "$prefix" | sed 's/_[0-9]\{8\}_[0-9]\{6\}$//' | sed 's/_[0-9]*B$//' | sed 's/_.*$//')
            configs+=("$config_name")
        fi
    done
    
    echo "Found ${#prefixes[@]} unique test prefixes:"
    for i in "${!prefixes[@]}"; do
        echo "  ✅ ${configs[$i]}: ${prefixes[$i]}"
    done
    
    # Apply config filtering if specified
    if [[ -n "$INCLUDE_CONFIGS" ]]; then
        echo ""
        echo "🔍 Filtering to include only: $INCLUDE_CONFIGS"
        
        IFS=',' read -ra INCLUDE_ARRAY <<< "$INCLUDE_CONFIGS"
        
        local filtered_prefixes=()
        local filtered_configs=()
        
        for i in "${!configs[@]}"; do
            local config="${configs[$i]}"
            local prefix="${prefixes[$i]}"
            
            local should_include=false
            for include_config in "${INCLUDE_ARRAY[@]}"; do
                include_config=$(echo "$include_config" | xargs)
                
                if [[ "$config" == *"$include_config"* ]] || [[ "$include_config" == *"$config"* ]]; then
                    should_include=true
                    break
                fi
            done
            
            if [[ "$should_include" == "true" ]]; then
                filtered_prefixes+=("$prefix")
                filtered_configs+=("$config")
                echo "  ✅ Including: $config ($prefix)"
            else
                echo "  ❌ Excluding: $config ($prefix)"
            fi
        done
        
        if [[ ${#filtered_prefixes[@]} -eq 0 ]]; then
            echo ""
            echo "❌ No configurations match the include filter: $INCLUDE_CONFIGS"
            exit 1
        fi
        
        prefixes=("${filtered_prefixes[@]}")
        configs=("${filtered_configs[@]}")
        
        echo ""
        echo "📊 Filtered to ${#prefixes[@]} configurations"
    fi
    
    echo ""
    echo "📊 Analyzing selected test results"
    echo "Configurations found: ${configs[*]}"
    echo ""
    
    # Run analysis
    CONFIGS_STRING=$(IFS=','; echo "${configs[*]}")
    PREFIXES_STRING=$(IFS=','; echo "${prefixes[*]}")
    python3 /tmp/analyze_long_doc_qa.py "$RESULTS_DIR" "$PREFIXES_STRING" "$CONFIGS_STRING" "$BASELINE_CONFIG" "$SHOW_IMPROVEMENTS"
    
    echo ""
    echo "📁 Raw summary files location: $RESULTS_DIR"
    echo "Files analyzed:"
    for i in "${!prefixes[@]}"; do
        echo "  ${configs[$i]} (${prefixes[$i]}):"
        ls -la "${RESULTS_DIR}/${prefixes[$i]}_summary.csv" 2>/dev/null || echo "    No file found"
    done
    
    exit 0
}

# Check if we should analyze all files
if [[ "$ANALYZE_ALL" == "true" ]]; then
    analyze_all_files
fi

# Try to find configuration files
BASELINE_TIMESTAMP=$(find_latest_config "baseline")
PRODUCTION_TIMESTAMP=$(find_latest_config "production")
CONNECTOR_TIMESTAMP=$(find_latest_config "connector")
LMCACHE_TIMESTAMP=$(find_latest_config "lmcache")

# If standard names don't work, try finding with prefixes
if [[ -z "$BASELINE_TIMESTAMP" ]]; then
    BASELINE_PREFIX=$(find_config_with_prefix "baseline")
    if [[ -n "$BASELINE_PREFIX" ]]; then
        BASELINE_TIMESTAMP="found"
    fi
fi

if [[ -z "$PRODUCTION_TIMESTAMP" ]]; then
    PRODUCTION_PREFIX=$(find_config_with_prefix "production")
    if [[ -n "$PRODUCTION_PREFIX" ]]; then
        PRODUCTION_TIMESTAMP="found"
    fi
fi

if [[ -z "$CONNECTOR_TIMESTAMP" ]]; then
    CONNECTOR_PREFIX=$(find_config_with_prefix "connector")
    if [[ -n "$CONNECTOR_PREFIX" ]]; then
        CONNECTOR_TIMESTAMP="found"
    fi
fi

if [[ -z "$LMCACHE_TIMESTAMP" ]]; then
    LMCACHE_PREFIX=$(find_config_with_prefix "lmcache")
    if [[ -n "$LMCACHE_PREFIX" ]]; then
        LMCACHE_TIMESTAMP="found"
    fi
fi

FOUND_CONFIGS=()
CONFIG_PREFIXES=()

if [[ -n "$BASELINE_TIMESTAMP" ]]; then
    FOUND_CONFIGS+=("baseline")
    if [[ -n "$BASELINE_PREFIX" ]]; then
        CONFIG_PREFIXES+=("$BASELINE_PREFIX")
        echo "✅ Found baseline: $BASELINE_PREFIX"
    else
        # Try to find the full prefix with model size
        FULL_PREFIX=$(find "$RESULTS_DIR" -name "baseline_*${BASELINE_TIMESTAMP}_summary.csv" 2>/dev/null | head -1 | xargs basename | sed 's/_summary.csv$//')
        CONFIG_PREFIXES+=("$FULL_PREFIX")
        echo "✅ Found baseline: $FULL_PREFIX"
    fi
fi

if [[ -n "$PRODUCTION_TIMESTAMP" ]]; then
    FOUND_CONFIGS+=("production")
    if [[ -n "$PRODUCTION_PREFIX" ]]; then
        CONFIG_PREFIXES+=("$PRODUCTION_PREFIX")
        echo "✅ Found production: $PRODUCTION_PREFIX"
    else
        FULL_PREFIX=$(find "$RESULTS_DIR" -name "production_*${PRODUCTION_TIMESTAMP}_summary.csv" 2>/dev/null | head -1 | xargs basename | sed 's/_summary.csv$//')
        CONFIG_PREFIXES+=("$FULL_PREFIX")
        echo "✅ Found production: $FULL_PREFIX"
    fi
fi

if [[ -n "$CONNECTOR_TIMESTAMP" ]]; then
    FOUND_CONFIGS+=("connector")
    if [[ -n "$CONNECTOR_PREFIX" ]]; then
        CONFIG_PREFIXES+=("$CONNECTOR_PREFIX")
        echo "✅ Found connector: $CONNECTOR_PREFIX"
    else
        FULL_PREFIX=$(find "$RESULTS_DIR" -name "connector_*${CONNECTOR_TIMESTAMP}_summary.csv" 2>/dev/null | head -1 | xargs basename | sed 's/_summary.csv$//')
        CONFIG_PREFIXES+=("$FULL_PREFIX")
        echo "✅ Found connector: $FULL_PREFIX"
    fi
fi

if [[ -n "$LMCACHE_TIMESTAMP" ]]; then
    FOUND_CONFIGS+=("lmcache")
    if
