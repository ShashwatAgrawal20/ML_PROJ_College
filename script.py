import warnings
from rich.table import Table
from rich.progress import Progress
from rich.console import Console
from rich.panel import Panel
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

warnings.filterwarnings('ignore')
console = Console()


def preprocess(dataframe, is_train=True):
    """Preprocesses the dataframe by encoding categorical features and scaling numerical features."""
    console.log("Starting data preprocessing...")
    cat_cols = ['protocol_type', 'service', 'flag']
    num_cols = [
        col for col in dataframe.columns if col not in cat_cols + ['outcome', 'level']]

    if is_train:
        console.log("Fitting OneHotEncoder on categorical columns...")
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoded_df = pd.DataFrame(encoder.fit_transform(dataframe[cat_cols]))
        encoded_df.columns = encoder.get_feature_names_out(cat_cols)
        joblib.dump(encoder, "encoder.pkl")
    else:
        console.log("Loading and applying pre-fitted OneHotEncoder...")
        encoder = joblib.load("encoder.pkl")
        encoded_df = pd.DataFrame(encoder.transform(dataframe[cat_cols]))
        encoded_df.columns = encoder.get_feature_names_out(cat_cols)

    dataframe = dataframe.drop(cat_cols, axis=1).reset_index(drop=True)
    dataframe = pd.concat([dataframe, encoded_df], axis=1)

    if num_cols:
        if is_train:
            console.log("Fitting RobustScaler on numerical columns...")
            scaler = RobustScaler()
            dataframe[num_cols] = scaler.fit_transform(dataframe[num_cols])
            joblib.dump(scaler, "scaler.pkl")
        else:
            console.log("Loading and applying pre-fitted RobustScaler...")
            scaler = joblib.load("scaler.pkl")
            try:
                present_cols = [c for c in num_cols if c in dataframe.columns]
                dataframe[present_cols] = scaler.transform(
                    dataframe[present_cols])
            except Exception as e:
                console.log(f"[bold red]Error during scaling: {e}[/bold red]")

    console.log("Preprocessing complete.")
    return dataframe


def plot_feature_importance(models, X_train, top_n=15):
    """Plot feature importance from tree-based models."""
    console.log(
        "[bold cyan]Generating feature importance analysis...[/bold cyan]")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Feature Importance Analysis Across Models',
                 fontsize=16, fontweight='bold')

    model_list = [
        ("Random Forest", models['Random Forest']),
        ("XGBoost", models['XGBoost']),
        ("Decision Tree", models['Decision Tree'])
    ]

    for idx, (name, model) in enumerate(model_list):
        ax = axes[idx // 2, idx % 2]

        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[-top_n:]

            ax.barh(range(top_n), importances[indices], color='steelblue')
            ax.set_yticks(range(top_n))
            ax.set_yticklabels([X_train.columns[i]
                               for i in indices], fontsize=8)
            ax.set_xlabel('Importance', fontsize=10)
            ax.set_title(f'{name}', fontsize=12, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)

    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig('feature_importance_analysis.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    console.log(
        "[green]Feature importance plot saved as 'feature_importance_analysis.png'[/green]")


def plot_model_comparison(performance_metrics):
    """Create a comprehensive model comparison visualization."""
    console.log(
        "[bold cyan]Generating model performance comparison...[/bold cyan]")

    df_metrics = pd.DataFrame(performance_metrics).T

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Performance Comparison Dashboard',
                 fontsize=16, fontweight='bold')

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']

    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        values = df_metrics[metric].values
        models = df_metrics.index.tolist()

        bars = ax.barh(models, values, color=colors[idx], alpha=0.8)
        ax.set_xlabel(metric, fontsize=11, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.grid(axis='x', alpha=0.3)

        # Add value labels on bars
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2,
                    f'{width:.3f}', ha='left', va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig('model_comparison_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    console.log(
        "[green]Model comparison saved as 'model_comparison_dashboard.png'[/green]")


def create_ensemble_model(models):
    """Create an ensemble voting classifier from trained models."""
    console.log(
        "[bold cyan]Creating Ensemble Voting Classifier...[/bold cyan]")

    estimators = [
        ('dt', models['Decision Tree']),
        ('rf', models['Random Forest']),
        ('xgb', models['XGBoost']),
        ('mlp', models['Neural Network (MLP)'])
    ]

    ensemble = VotingClassifier(estimators=estimators, voting='soft')
    console.log("[green]Ensemble model created with soft voting![/green]")
    return ensemble


def train_and_evaluate(data_path):
    """Loads data, preprocesses it, and trains multiple classification models."""
    console.log("Loading data from file...")
    columns = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
        'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
        'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
        'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
        'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
        'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
        'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'outcome', 'level'
    ]
    data = pd.read_csv(data_path, names=columns, header=None)
    data['outcome'] = data['outcome'].apply(
        lambda x: 0 if x == 'normal' else 1)

    processed_data = preprocess(data, is_train=True)
    train_cols = [
        col for col in processed_data.columns if col not in ['outcome', 'level']]
    joblib.dump(train_cols, 'train_cols.pkl')

    X = processed_data[train_cols]
    y = processed_data['outcome']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1),
        "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42, early_stopping=True)
    }

    performance_metrics = {}

    with Progress() as progress:
        train_task = progress.add_task(
            "[green]Training models...", total=len(models))
        for name, model in models.items():
            console.log(f"--- Training {name} ---")
            model.fit(X_train, y_train)
            joblib.dump(model, f"{name.lower().replace(
                ' ', '_').replace('(', '').replace(')', '')}_model.pkl")

            console.log(f"Evaluating {name}...")
            predictions = model.predict(X_test)

            performance_metrics[name] = {
                'Accuracy': accuracy_score(y_test, predictions),
                'Precision': precision_score(y_test, predictions),
                'Recall': recall_score(y_test, predictions),
                'F1-Score': f1_score(y_test, predictions)
            }

            console.print(f"[bold yellow]{
                          name} Classification Report:[/bold yellow]")
            console.print(classification_report(
                y_test, predictions, target_names=['Normal', 'Attack']))

            cm = confusion_matrix(y_test, predictions)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[
                        'Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
            plt.title(f'{name} Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.savefig(f"{name.lower().replace(' ', '_').replace(
                '(', '').replace(')', '')}_confusion_matrix.png")
            plt.close()

            progress.update(train_task, advance=1)

    console.log(
        "[bold cyan]Creating and evaluating Ensemble Model...[/bold cyan]")
    ensemble = create_ensemble_model(models)
    ensemble.fit(X_train, y_train)
    ensemble_predictions = ensemble.predict(X_test)

    performance_metrics["Ensemble (Voting)"] = {
        'Accuracy': accuracy_score(y_test, ensemble_predictions),
        'Precision': precision_score(y_test, ensemble_predictions),
        'Recall': recall_score(y_test, ensemble_predictions),
        'F1-Score': f1_score(y_test, ensemble_predictions)
    }

    joblib.dump(ensemble, "ensemble_voting_model.pkl")
    console.log("[green]Ensemble model saved![/green]")

    plot_feature_importance(models, X_train)
    plot_model_comparison(performance_metrics)

    console.log(
        "[bold green]All models trained and saved successfully![/bold green]")


def prediction_cli():
    """Interactive CLI with ensemble predictions and confidence analysis."""
    console.log(
        "[bold cyan]Loading models and preprocessors for prediction...[/bold cyan]")

    models = {
        "Decision Tree": joblib.load("decision_tree_model.pkl"),
        "Random Forest": joblib.load("random_forest_model.pkl"),
        "XGBoost": joblib.load("xgboost_model.pkl"),
        "Neural Network (MLP)": joblib.load("neural_network_mlp_model.pkl")
    }

    try:
        ensemble = joblib.load("ensemble_voting_model.pkl")
        models["Ensemble (Voting)"] = ensemble
        console.log("[green]Ensemble model loaded successfully![/green]")
    except:
        console.log(
            "[yellow]Ensemble model not found, fitting it now (one-time process)...[/yellow]")
        # Need to load training data to fit ensemble
        try:
            columns = [
                'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
                'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
                'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
                'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
                'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
                'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
                'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
                'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
                'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
                'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'outcome', 'level'
            ]
            data = pd.read_csv("nsl-kdd/KDDTrain+.txt",
                               names=columns, header=None)
            data['outcome'] = data['outcome'].apply(
                lambda x: 0 if x == 'normal' else 1)
            processed_data = preprocess(data, is_train=False)
            X = processed_data[train_cols]
            y = processed_data['outcome']

            # Sample data for faster fitting (use 10% of data)
            sample_size = min(5000, len(X))
            indices = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X.iloc[indices]
            y_sample = y.iloc[indices]

            ensemble = create_ensemble_model(models)
            console.log(
                "[cyan]Fitting ensemble model on sample data...[/cyan]")
            ensemble.fit(X_sample, y_sample)
            joblib.dump(ensemble, "ensemble_voting_model.pkl")
            models["Ensemble (Voting)"] = ensemble
            console.log("[green]Ensemble fitted and saved![/green]")
        except Exception as e:
            console.log(f"[red]Could not fit ensemble: {e}[/red]")
            console.log(
                "[yellow]Continuing without ensemble model...[/yellow]")

    train_cols = joblib.load('train_cols.pkl')

    predefined_data = [
        {'duration': 0, 'protocol_type': 'tcp', 'service': 'http', 'flag': 'SF', 'src_bytes': 300, 'dst_bytes': 2500, 'land': 0, 'wrong_fragment': 0, 'urgent': 0, 'hot': 0, 'num_failed_logins': 0, 'logged_in': 1, 'num_compromised': 0, 'root_shell': 0, 'su_attempted': 0, 'num_root': 0, 'num_file_creations': 0, 'num_shells': 0, 'num_access_files': 0, 'num_outbound_cmds': 0, 'is_host_login': 0, 'is_guest_login': 0, 'count': 4, 'srv_count': 4, 'serror_rate': 0.0,
            'srv_serror_rate': 0.0, 'rerror_rate': 0.0, 'srv_rerror_rate': 0.0, 'same_srv_rate': 1.0, 'diff_srv_rate': 0.0, 'srv_diff_host_rate': 0.0, 'dst_host_count': 150, 'dst_host_srv_count': 255, 'dst_host_same_srv_rate': 1.0, 'dst_host_diff_srv_rate': 0.0, 'dst_host_same_src_port_rate': 0.01, 'dst_host_srv_diff_host_rate': 0.0, 'dst_host_serror_rate': 0.0, 'dst_host_srv_serror_rate': 0.0, 'dst_host_rerror_rate': 0.0, 'dst_host_srv_rerror_rate': 0.0},
        {'duration': 2, 'protocol_type': 'tcp', 'service': 'ftp_data', 'flag': 'SF', 'src_bytes': 1500, 'dst_bytes': 0, 'land': 0, 'wrong_fragment': 0, 'urgent': 0, 'hot': 0, 'num_failed_logins': 0, 'logged_in': 1, 'num_compromised': 0, 'root_shell': 0, 'su_attempted': 0, 'num_root': 0, 'num_file_creations': 0, 'num_shells': 0, 'num_access_files': 0, 'num_outbound_cmds': 0, 'is_host_login': 0, 'is_guest_login': 0, 'count': 1, 'srv_count': 1, 'serror_rate': 0.0,
            'srv_serror_rate': 0.0, 'rerror_rate': 0.0, 'srv_rerror_rate': 0.0, 'same_srv_rate': 1.0, 'diff_srv_rate': 0.0, 'srv_diff_host_rate': 0.0, 'dst_host_count': 20, 'dst_host_srv_count': 30, 'dst_host_same_srv_rate': 0.5, 'dst_host_diff_srv_rate': 0.1, 'dst_host_same_src_port_rate': 1.0, 'dst_host_srv_diff_host_rate': 0.0, 'dst_host_serror_rate': 0.0, 'dst_host_srv_serror_rate': 0.0, 'dst_host_rerror_rate': 0.0, 'dst_host_srv_rerror_rate': 0.0},
        {'duration': 0, 'protocol_type': 'tcp', 'service': 'private', 'flag': 'S0', 'src_bytes': 0, 'dst_bytes': 0, 'land': 0, 'wrong_fragment': 0, 'urgent': 0, 'hot': 0, 'num_failed_logins': 0, 'logged_in': 0, 'num_compromised': 0, 'root_shell': 0, 'su_attempted': 0, 'num_root': 0, 'num_file_creations': 0, 'num_shells': 0, 'num_access_files': 0, 'num_outbound_cmds': 0, 'is_host_login': 0, 'is_guest_login': 0, 'count': 200, 'srv_count': 10, 'serror_rate': 1.00,
            'srv_serror_rate': 1.00, 'rerror_rate': 0.00, 'srv_rerror_rate': 0.00, 'same_srv_rate': 0.05, 'diff_srv_rate': 0.06, 'srv_diff_host_rate': 0.00, 'dst_host_count': 255, 'dst_host_srv_count': 10, 'dst_host_same_srv_rate': 0.04, 'dst_host_diff_srv_rate': 0.06, 'dst_host_same_src_port_rate': 0.00, 'dst_host_srv_diff_host_rate': 0.00, 'dst_host_serror_rate': 1.00, 'dst_host_srv_serror_rate': 1.00, 'dst_host_rerror_rate': 0.00, 'dst_host_srv_rerror_rate': 0.00},
        {'duration': 0, 'protocol_type': 'tcp', 'service': 'eco_i', 'flag': 'SF', 'src_bytes': 8, 'dst_bytes': 0, 'land': 0, 'wrong_fragment': 0, 'urgent': 0, 'hot': 0, 'num_failed_logins': 0, 'logged_in': 0, 'num_compromised': 0, 'root_shell': 0, 'su_attempted': 0, 'num_root': 0, 'num_file_creations': 0, 'num_shells': 0, 'num_access_files': 0, 'num_outbound_cmds': 0, 'is_host_login': 0, 'is_guest_login': 0, 'count': 500, 'srv_count': 500, 'serror_rate': 0.0,
            'srv_serror_rate': 0.0, 'rerror_rate': 0.0, 'srv_rerror_rate': 0.0, 'same_srv_rate': 1.0, 'diff_srv_rate': 0.0, 'srv_diff_host_rate': 0.0, 'dst_host_count': 255, 'dst_host_srv_count': 200, 'dst_host_same_srv_rate': 0.8, 'dst_host_diff_srv_rate': 0.1, 'dst_host_same_src_port_rate': 0.8, 'dst_host_srv_diff_host_rate': 0.0, 'dst_host_serror_rate': 0.0, 'dst_host_srv_serror_rate': 0.0, 'dst_host_rerror_rate': 0.0, 'dst_host_srv_rerror_rate': 0.0},
        {'duration': 4, 'protocol_type': 'tcp', 'service': 'ftp', 'flag': 'SF', 'src_bytes': 150, 'dst_bytes': 200, 'land': 0, 'wrong_fragment': 0, 'urgent': 0, 'hot': 3, 'num_failed_logins': 5, 'logged_in': 0, 'num_compromised': 0, 'root_shell': 0, 'su_attempted': 0, 'num_root': 0, 'num_file_creations': 0, 'num_shells': 0, 'num_access_files': 0, 'num_outbound_cmds': 0, 'is_host_login': 0, 'is_guest_login': 1, 'count': 1, 'srv_count': 1, 'serror_rate': 0.0,
            'srv_serror_rate': 0.0, 'rerror_rate': 0.0, 'srv_rerror_rate': 0.0, 'same_srv_rate': 1.0, 'diff_srv_rate': 0.0, 'srv_diff_host_rate': 0.0, 'dst_host_count': 10, 'dst_host_srv_count': 2, 'dst_host_same_srv_rate': 0.2, 'dst_host_diff_srv_rate': 0.8, 'dst_host_same_src_port_rate': 1.0, 'dst_host_srv_diff_host_rate': 0.0, 'dst_host_serror_rate': 0.0, 'dst_host_srv_serror_rate': 0.0, 'dst_host_rerror_rate': 0.0, 'dst_host_srv_rerror_rate': 0.0},
        {'duration': 0, 'protocol_type': 'icmp', 'service': 'ecr_i', 'flag': 'SF', 'src_bytes': 1032, 'dst_bytes': 0, 'land': 0, 'wrong_fragment': 0, 'urgent': 0, 'hot': 0, 'num_failed_logins': 0, 'logged_in': 0, 'num_compromised': 0, 'root_shell': 0, 'su_attempted': 0, 'num_root': 0, 'num_file_creations': 0, 'num_shells': 0, 'num_access_files': 0, 'num_outbound_cmds': 0, 'is_host_login': 0, 'is_guest_login': 0, 'count': 511, 'srv_count': 511, 'serror_rate': 0.0,
            'srv_serror_rate': 0.0, 'rerror_rate': 0.0, 'srv_rerror_rate': 0.0, 'same_srv_rate': 1.0, 'diff_srv_rate': 0.0, 'srv_diff_host_rate': 0.0, 'dst_host_count': 255, 'dst_host_srv_count': 255, 'dst_host_same_srv_rate': 1.0, 'dst_host_diff_srv_rate': 0.0, 'dst_host_same_src_port_rate': 1.0, 'dst_host_srv_diff_host_rate': 0.0, 'dst_host_serror_rate': 0.0, 'dst_host_srv_serror_rate': 0.0, 'dst_host_rerror_rate': 0.0, 'dst_host_srv_rerror_rate': 0.0}
    ]

    scenario_descriptions = [
        "Normal HTTP Traffic (Benign)",
        "Normal FTP Traffic (Benign)",
        "SYN Flood DoS Attack",
        "Port Scan / Nmap Probe",
        "FTP Brute Force Attack",
        "ICMP Smurf Attack (DoS)"
    ]

    while True:
        console.print("\n" + "="*60)
        console.print(Panel("[bold cyan]Enhanced IDS - Intrusion Detection with Ensemble AI[/bold cyan]",
                            style="bold white on blue"))
        console.print("="*60)
        console.print(
            f"[yellow]Select a network traffic scenario (1-{len(predefined_data)}):[/yellow]")

        for i, desc in enumerate(scenario_descriptions, 1):
            color = "green" if i <= 2 else "red"
            console.print(f"  [{color}]{i}. {desc}[/{color}]")

        console.print("\n  [cyan]0. Exit[/cyan]")
        console.print("="*60)

        user_input = input("Your choice: ")
        try:
            choice = int(user_input)
        except ValueError:
            console.print("[red]Invalid input. Please enter a number.[/red]")
            continue

        if choice == 0:
            console.log(
                "[bold cyan]Exiting Enhanced IDS. Stay safe![/bold cyan]")
            break

        if 1 <= choice <= len(predefined_data):
            console.log(f"[cyan]Analyzing scenario {choice}: {
                        scenario_descriptions[choice-1]}...[/cyan]")
            test_df = pd.DataFrame([predefined_data[choice - 1]])
            processed_test_df = preprocess(test_df, is_train=False)
            processed_test_df = processed_test_df.reindex(
                columns=train_cols, fill_value=0)

            table = Table(title=f"🔍 Detection Results for Scenario {choice}",
                          title_style="bold magenta",
                          show_header=True,
                          header_style="bold cyan")
            table.add_column("Model", style="cyan", no_wrap=True, width=25)
            table.add_column("Prediction", style="magenta", width=15)
            table.add_column("Confidence", style="yellow", width=12)
            table.add_column("Risk Level", style="white", width=15)

            predictions_list = []

            for name, model in models.items():
                probs = model.predict_proba(processed_test_df)[0]
                prediction = np.argmax(probs)
                confidence = probs[prediction]
                predictions_list.append(prediction)

                label = "🚨 Attack" if prediction == 1 else "✅ Normal"
                style = "bold red" if prediction == 1 else "bold green"
                confidence_str = f"{confidence:.2%}"

                # Risk level based on confidence
                if prediction == 1:
                    if confidence > 0.9:
                        risk = "🔴 Critical"
                    elif confidence > 0.7:
                        risk = "🟠 High"
                    else:
                        risk = "🟡 Medium"
                else:
                    risk = "🟢 Safe"

                table.add_row(name, f"[{style}]{
                              label}[/{style}]", confidence_str, risk)

            console.print(table)

            # Consensus analysis
            attack_count = sum(predictions_list)
            consensus = attack_count / len(predictions_list)

            console.print("\n" + "="*60)
            if consensus >= 0.6:
                console.print(Panel(
                    f"[bold red]✅ THREAT DETECTED![/bold red]\n"
                    f"Consensus: {
                        attack_count}/{len(predictions_list)} models detected an attack\n"
                    f"Confidence: {consensus*100:.1f}%\n"
                    f"Recommendation: Block traffic and investigate immediately",
                    style="bold white on red"
                ))
            else:
                console.print(Panel(
                    f"[bold green]✅ TRAFFIC APPEARS NORMAL[/bold green]\n"
                    f"Consensus: {
                        len(predictions_list)-attack_count}/{len(predictions_list)} models approve\n"
                    f"Confidence: {(1-consensus)*100:.1f}%\n"
                    f"Recommendation: Allow traffic, continue monitoring",
                    style="bold white on green"
                ))
            console.print("="*60 + "\n")

        else:
            console.print(
                f"[red]Invalid choice. Please select 1-{len(predefined_data)} or 0 to exit.[/red]")


if __name__ == "__main__":
    console.print(Panel.fit(
        "[bold cyan]🛡️  Enhanced Intrusion Detection System (IDS) with AI Ensemble[/bold cyan]\n"
        "[yellow]Featuring: Ensemble Learning, Feature Analysis & Explainable AI[/yellow]",
        border_style="bold blue"
    ))

    try:
        joblib.load("random_forest_model.pkl")
        console.log("[green]✓ Found existing models, loading them...[/green]")
    except FileNotFoundError:
        console.log(
            "[yellow]⚠ No models found, starting training process...[/yellow]")
        train_and_evaluate("nsl-kdd/KDDTrain+.txt")

    prediction_cli()
