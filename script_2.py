import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from rich.console import Console
from rich.progress import Progress

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Create a console object
console = Console()

# --- Preprocessing Function ---


def preprocess_multiclass(dataframe, is_train=True, label_encoder=None):
    """
    Preprocesses data for multi-class classification.
    Handles categorical features, numerical scaling, and target variable encoding.
    """
    console.log("Starting multi-class data preprocessing...")
    cat_cols = ['protocol_type', 'service', 'flag']
    num_cols = [
        col for col in dataframe.columns if col not in cat_cols + ['outcome', 'level']]

    # --- Target Variable Mapping ---
    # Group detailed attack types into 5 broader categories
    attack_mapping = {
        'normal': 'normal',
        'neptune': 'dos', 'smurf': 'dos', 'pod': 'dos', 'teardrop': 'dos', 'land': 'dos', 'back': 'dos',
        'satan': 'probe', 'ipsweep': 'probe', 'portsweep': 'probe', 'nmap': 'probe',
        'warezclient': 'r2l', 'guess_passwd': 'r2l', 'ftp_write': 'r2l', 'multihop': 'r2l', 'phf': 'r2l', 'imap': 'r2l', 'warezmaster': 'r2l', 'spy': 'r2l',
        'buffer_overflow': 'u2r', 'loadmodule': 'u2r', 'perl': 'u2r', 'rootkit': 'u2r'
    }
    dataframe['outcome'] = dataframe['outcome'].map(attack_mapping).fillna(
        'other')  # Map and handle any unlisted attacks

    # --- Encode Target Variable ---
    if is_train:
        console.log("Fitting LabelEncoder on the 'outcome' column...")
        le = LabelEncoder()
        dataframe['outcome'] = le.fit_transform(dataframe['outcome'])
        joblib.dump(le, "multiclass_label_encoder.pkl")
    else:
        if label_encoder:
            # Handle unseen labels during prediction
            dataframe['outcome'] = dataframe['outcome'].apply(
                lambda x: x if x in label_encoder.classes_ else 'other')
            dataframe['outcome'] = label_encoder.transform(
                dataframe['outcome'])

    # --- Handle Feature Columns (as in binary script) ---
    if is_train:
        console.log("Fitting OneHotEncoder on categorical features...")
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoded_df = pd.DataFrame(encoder.fit_transform(dataframe[cat_cols]))
        encoded_df.columns = encoder.get_feature_names_out(cat_cols)
        joblib.dump(encoder, "multiclass_encoder.pkl")
    else:
        console.log("Loading and applying pre-fitted OneHotEncoder...")
        encoder = joblib.load("multiclass_encoder.pkl")
        encoded_df = pd.DataFrame(encoder.transform(dataframe[cat_cols]))
        encoded_df.columns = encoder.get_feature_names_out(cat_cols)

    dataframe = dataframe.drop(cat_cols, axis=1).reset_index(drop=True)
    dataframe = pd.concat([dataframe, encoded_df], axis=1)

    if num_cols:
        if is_train:
            console.log("Fitting RobustScaler on numerical columns...")
            scaler = RobustScaler()
            dataframe[num_cols] = scaler.fit_transform(dataframe[num_cols])
            joblib.dump(scaler, "multiclass_scaler.pkl")
        else:
            console.log("Loading and applying pre-fitted RobustScaler...")
            scaler = joblib.load("multiclass_scaler.pkl")
            train_cols = scaler.get_feature_names_out()
            current_num_cols = [
                col for col in train_cols if col in dataframe.columns]
            dataframe[current_num_cols] = scaler.transform(
                dataframe[current_num_cols])

    console.log("Preprocessing complete.")
    return dataframe

# --- Model Training and Evaluation for Multi-Class ---


def train_and_evaluate_multiclass(data_path):
    """
    Trains and evaluates models for multi-class intrusion detection.
    """
    console.log("--- Starting Multi-Class Model Training ---")
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

    # Preprocess data
    data = preprocess_multiclass(data, is_train=True)
    label_encoder = joblib.load("multiclass_label_encoder.pkl")
    class_names = label_encoder.classes_

    train_cols = [
        col for col in data.columns if col not in ['outcome', 'level']]
    joblib.dump(train_cols, 'multiclass_train_cols.pkl')  # Save column order

    X = data[train_cols]
    y = data['outcome']

    console.log(
        "Splitting multi-class data into training (80%) and test (20%) sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, n_jobs=-1)
    }

    with Progress() as progress:
        train_task = progress.add_task(
            "[green]Training multi-class models...", total=len(models))
        for name, model in models.items():
            console.log(
                f"--- Training {name} for Multi-Class Classification ---")
            model.fit(X_train, y_train)
            joblib.dump(model, f"multiclass_{
                        name.lower().replace(' ', '_')}_model.pkl")

            console.log(f"Evaluating {name}...")
            predictions = model.predict(X_test)

            console.print(f"[bold yellow]{
                          name} Multi-Class Classification Report:[/bold yellow]")
            console.print(classification_report(
                y_test, predictions, target_names=class_names))

            # Generate and Save Confusion Matrix
            cm = confusion_matrix(y_test, predictions)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='viridis',
                        xticklabels=class_names, yticklabels=class_names)
            plt.title(f'{name} Multi-Class Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.tight_layout()
            plt.savefig(f"multiclass_{name.lower().replace(
                ' ', '_')}_confusion_matrix.png")
            plt.close()
            console.log(f"Multi-class confusion matrix for {name} saved.")

            progress.update(train_task, advance=1)

    console.log(
        "[bold green]All multi-class models trained and saved successfully![/bold green]")


if __name__ == "__main__":
    train_and_evaluate_multiclass("nsl-kdd/KDDTrain+.txt")
    console.log("\n[bold]To make predictions, you would typically build a CLI similar to the binary script, but using the saved 'multiclass' models.[/bold]")
