import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def generate_dashboard(session_dir):
    session_dir = Path(session_dir)
    csv_path = session_dir / "events.csv"
    
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print("Error: Session log is empty.")
        return

    # Convert timestamp to relative time (seconds)
    df['time_rel'] = df['timestamp'] - df['timestamp'].iloc[0]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1]})
    
    # 1. EAR (alertness proxy) over time
    ear_df = df[df['event_type'] == 'EAR']
    ax1.plot(ear_df['time_rel'], ear_df['value'], color='#3498db', label='EAR (Eye Aspect Ratio)')
    ax1.set_title('Driver Alertness (EAR) Over Time')
    ax1.set_ylabel('EAR Value')
    ax1.axhline(y=0.2, color='red', linestyle='--', label='Drowsy Threshold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Alert Occurrences
    alerts_df = df[df['event_type'] == 'ALERT']
    if not alerts_df.empty:
        alert_types = alerts_df['value'].value_counts()
        alert_types.plot(kind='bar', ax=ax2, color=['#e74c3c', '#f39c12', '#9b59b6', '#2ecc71'])
        ax2.set_title('Alert Frequency Summary')
        ax2.set_ylabel('Count')
    else:
        ax2.text(0.5, 0.5, 'No alerts triggered during session', ha='center', va='center')
        ax2.set_title('Alert Frequency Summary')

    plt.tight_layout()
    output_path = session_dir / "analytics_summary.png"
    plt.savefig(output_path, dpi=300)
    print(f"Dashboard saved to {output_path}")
    plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("session_dir", help="Path to the session log directory")
    args = parser.parse_args()
    
    generate_dashboard(args.session_dir)
