import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import warnings
import scipy.io as sio
from torch.utils.data import DataLoader, TensorDataset
from Index_calculation import *
import datetime
import math
import random
import scipy.stats
import time
from net_lab import CSS

warnings.filterwarnings("ignore")

# --- Global Configuration ---
RANDOM_SEED = 42
BATCH_SIZE = 64
LEARNING_RATE = 0.0005
EPOCHS = 300
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- JSD ---
NUM_SELECTED_SOURCES = 10

# --- Beta Dynamic Configuration---
BETA_START = 0.0
BETA_END = 0.5
BETA_WARMUP_EPOCHS = 50

SEQUENCE_LENGTH = 10

# --- Early Stopping Configuration ---
PATIENCE = 30
MIN_DELTA = 0.001
STARTEPOCH = 50

# --- Alignment Loss Weight ---
LAMBDA_ALIGN = 0.5

# --- NSAL Configuration ---
USE_NSAL = True
NSAL_K = 9
NSAL_WARMUP_EPOCHS = 20
NSAL_WEIGHT_START = 0.0
NSAL_WEIGHT_END = 0.1

EEG_FEATURE_PATH = "./SADT_EEG_Features_DE"
LABEL_PATH = "./SADT_labels"
# EEG_FEATURE_PATH = "./SEED_VIG_EEG_Features_DE"
# LABEL_PATH = "./SEED_VIG_labels"
# Define the number of subjects for the SADT dataset
NUM_SUBJECTS = 11


def calculate_jsd(p_features, q_features, bins=256):
    p_flat, q_flat = p_features.reshape(-1), q_features.reshape(-1)
    p_hist = np.histogram(p_flat, bins=bins, density=True)[0] + 1e-10
    q_hist = np.histogram(q_flat, bins=bins, density=True)[0] + 1e-10
    m_hist = (p_hist + q_hist) / 2
    return (scipy.stats.entropy(p_hist, m_hist) + scipy.stats.entropy(q_hist, m_hist)) / 2


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_sequences(features, labels, seq_length):
    sequences, sequence_labels = [], []
    num_samples = features.shape[0]
    for i in range(num_samples - seq_length + 1):
        sequence = features[i: i + seq_length]
        label = labels[i + seq_length - 1]
        sequences.append(sequence)
        sequence_labels.append(label)
    return np.array(sequences), np.array(sequence_labels)


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop


def get_progressive_beta(epoch, total_epochs):

    if epoch < BETA_WARMUP_EPOCHS:

        return BETA_START
    else:

        progress = min(1.0, (epoch - BETA_WARMUP_EPOCHS) / (total_epochs - BETA_WARMUP_EPOCHS))
        return BETA_START + (BETA_END - BETA_START) * progress


if __name__ == '__main__':
    start_time = time.time()
    set_seed(RANDOM_SEED)
    log_filename = f"training_log_Aligned_Temporal_Strategy1_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(log_filename, 'w') as log_file:
        def log_and_print(message):
            print(message)
            #log_file.write(message + '\n')



        log_and_print(
            f"--- Experiment Configuration ---\nArchitecture: Shared (Temporal) + Unified Personalized Model for All Subjects + Feature Alignment\nRandom Seed: {RANDOM_SEED}\nBatch Size: {BATCH_SIZE}\nLearning Rate: {LEARNING_RATE}\nSequence Length: {SEQUENCE_LENGTH}\nAlignment Loss Weight (Lambda Align): {LAMBDA_ALIGN}")
        log_and_print(f"Adaptive Source Selection: Enabled (K={NUM_SELECTED_SOURCES})")
        log_and_print(
            f"Neighborhood Semantic Alignment Learning (NSAL): {'Enabled' if USE_NSAL else 'Disabled'} (K={NSAL_K}, Warmup={NSAL_WARMUP_EPOCHS}, Final Weight={NSAL_WEIGHT_END})")
        log_and_print(f"Beta Strategy: Progressive Strategy 1 (Linear Growth)")
        log_and_print(f"Beta Parameters: START={BETA_START}, END={BETA_END}, WARMUP_EPOCHS={BETA_WARMUP_EPOCHS}")
        log_and_print(f"Early Stopping: Enabled (Patience={PATIENCE}, Min Delta={MIN_DELTA}), STARTEPOCH={STARTEPOCH}")

        G = testclass()
        subject_features_raw, subject_labels_raw = [], []
        for i in range(1, NUM_SUBJECTS + 1):
            filename = f'{i}.mat'
            de_features_raw = sio.loadmat(os.path.join(EEG_FEATURE_PATH, filename))['de_LDS']
            current_features = de_features_raw.transpose(0, 2, 1)
            current_labels = np.load(os.path.join(LABEL_PATH, f"labels{i}.npy"))
            min_trials = min(current_features.shape[0], current_labels.shape[0])
            subject_features_raw.append(current_features[:min_trials])
            subject_labels_raw.append(current_labels[:min_trials])

        cross_validation_results = []
        all_subject_ids_list = list(range(NUM_SUBJECTS))

        for subject_id in range(1, NUM_SUBJECTS + 1):
            log_and_print(
                f"\n{'=' * 60}\n--- Starting Fold {subject_id}/{NUM_SUBJECTS}: Testing Subject {subject_id} ---\n{'=' * 60}")

            X_target_raw, Y_target_raw = subject_features_raw[subject_id - 1], subject_labels_raw[subject_id - 1]
            source_candidates_raw = [{'id': i, 'features': subject_features_raw[i], 'labels': subject_labels_raw[i]} for
                                     i in
                                     range(NUM_SUBJECTS) if (i + 1) != subject_id]

            jsd_scores = sorted([(calculate_jsd(X_target_raw, c['features']), c) for c in source_candidates_raw],
                                key=lambda x: x[0])
            selected_sources_raw = [score[1] for score in jsd_scores[:NUM_SELECTED_SOURCES]]
            selected_source_ids = [s['id'] for s in selected_sources_raw]
            log_and_print(
                f"Best source subject IDs (0-based) selected for target subject {subject_id}: {selected_source_ids}")

            source_x_list, source_y_list, source_id_list = [], [], []
            for s in selected_sources_raw:
                seq_x, seq_y = create_sequences(s['features'], s['labels'], SEQUENCE_LENGTH)
                source_x_list.append(seq_x)
                source_y_list.append(seq_y)
                source_id_list.append(np.full(seq_x.shape[0], s['id']))

            X_source, Y_source, IDS_source = np.vstack(source_x_list), np.vstack(source_y_list), np.hstack(
                source_id_list)

            X_target, Y_target = create_sequences(X_target_raw, Y_target_raw, SEQUENCE_LENGTH)
            target_subject_real_id = subject_id - 1
            IDS_target = np.full(X_target.shape[0], target_subject_real_id)

            if len(Y_target) == 0:
                log_and_print(
                    f"Target subject {subject_id} has insufficient data to create sequences of length {SEQUENCE_LENGTH}, skipping this fold.")
                continue

            source_dataset = TensorDataset(torch.FloatTensor(X_source), torch.FloatTensor(Y_source),
                                           torch.LongTensor(IDS_source))
            target_dataset = TensorDataset(torch.FloatTensor(X_target), torch.FloatTensor(Y_target),
                                           torch.LongTensor(IDS_target))

            source_dataloader = DataLoader(source_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
            target_dataloader = DataLoader(target_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

            myModel = CSS(all_subject_ids=all_subject_ids_list).to(DEVICE)

            loss_label_func, loss_domain_func = nn.MSELoss().to(DEVICE), nn.NLLLoss().to(DEVICE)
            opt = torch.optim.Adam(myModel.parameters(), lr=LEARNING_RATE)


            early_stopping = EarlyStopping(patience=PATIENCE, min_delta=MIN_DELTA)
            best_model_state = None
            best_epoch = 0
            best_beta_history = []

            for epoch in range(EPOCHS):
                myModel.train()
                len_dataloader = min(len(source_dataloader), len(target_dataloader))
                if len_dataloader == 0:
                    continue


                current_beta = get_progressive_beta(epoch, EPOCHS)
                best_beta_history.append(current_beta)

                iter_source, iter_target = iter(source_dataloader), iter(target_dataloader)
                gamma = NSAL_WEIGHT_START + (NSAL_WEIGHT_END - NSAL_WEIGHT_START) * (epoch / EPOCHS)

                epoch_losses = {
                    'total': 0.0,
                    'label': 0.0,
                    'domain': 0.0,
                    'consistency': 0.0,
                    'nsal': 0.0,
                    'alignment': 0.0
                }
                batch_count = 0

                for i in range(len_dataloader):
                    source_x, source_y, source_ids = next(iter_source)
                    target_x, _, target_ids = next(iter_target)
                    source_x, source_y, source_ids = source_x.to(DEVICE), source_y.to(DEVICE), source_ids.to(DEVICE)
                    target_x, target_ids = target_x.to(DEVICE), target_ids.to(DEVICE)


                    p = float(i + epoch * len_dataloader) / (EPOCHS * len_dataloader)
                    alpha = 2. / (1. + math.exp(-10 * p)) - 1

                    opt.zero_grad()

                    combined_x = torch.cat((source_x, target_x), 0)
                    combined_ids = torch.cat((source_ids, target_ids), 0)

                    label_outputs, domain_outputs, shared_features, loss_alignment = myModel(combined_x, combined_ids,
                                                                                             alpha)

                    source_label_out, target_label_out = label_outputs[:BATCH_SIZE], label_outputs[BATCH_SIZE:]
                    source_shared_features, target_shared_features = shared_features[:BATCH_SIZE], shared_features[
                                                                                                   BATCH_SIZE:]

                    loss_label = loss_label_func(source_label_out, source_y)
                    domain_label = torch.zeros(BATCH_SIZE * 2, dtype=torch.long, device=DEVICE)
                    domain_label[BATCH_SIZE:] = 1
                    loss_domain = loss_domain_func(domain_outputs, domain_label)

                    target_outputs2, _, _, _ = myModel(target_x, target_ids, alpha)
                    loss_consistency = loss_label_func(target_label_out, target_outputs2)

                    loss_nsal = torch.tensor(0.0).to(DEVICE)
                    if USE_NSAL and epoch > NSAL_WARMUP_EPOCHS:
                        with torch.no_grad():
                            source_feats_norm = F.normalize(source_shared_features)
                            target_feats_norm = F.normalize(target_shared_features)
                            distance_matrix = 1 - torch.mm(target_feats_norm, source_feats_norm.t())
                            distances, nn_indices = torch.topk(distance_matrix, NSAL_K, dim=1, largest=False)
                            weights = 1.0 / (distances + 1e-8)
                            weights = weights / torch.sum(weights, dim=1, keepdim=True)
                            source_labels_for_nsal = source_y[nn_indices]
                            pseudo_labels = torch.sum(source_labels_for_nsal * weights.unsqueeze(-1), dim=1)
                        loss_nsal = loss_label_func(target_label_out, pseudo_labels)


                    total_loss = loss_label + loss_domain + current_beta * loss_consistency + gamma * loss_nsal + LAMBDA_ALIGN * loss_alignment

                    total_loss.backward()
                    opt.step()


                    epoch_losses['total'] += total_loss.item()
                    epoch_losses['label'] += loss_label.item()
                    epoch_losses['domain'] += loss_domain.item()
                    epoch_losses['consistency'] += loss_consistency.item()
                    epoch_losses['nsal'] += loss_nsal.item()
                    epoch_losses['alignment'] += loss_alignment.item()
                    batch_count += 1


                avg_total_loss = epoch_losses['total'] / batch_count


                if epoch >= STARTEPOCH:
                    if early_stopping(avg_total_loss):
                        log_and_print(f"Early stopping triggered at epoch {epoch + 1}")
                        break


                if early_stopping.best_loss == avg_total_loss:
                    best_model_state = myModel.state_dict().copy()
                    best_epoch = epoch + 1

                log_and_print(f"Epoch: {epoch + 1}/{EPOCHS} | "
                              f"Beta: {current_beta:.4f} | "
                              f"AvgLoss: {avg_total_loss:.4f} (L:{epoch_losses['label'] / batch_count:.4f} D:{epoch_losses['domain'] / batch_count:.4f} "
                              f"C:{epoch_losses['consistency'] / batch_count:.4f} NSAL:{epoch_losses['nsal'] / batch_count:.4f} Align:{epoch_losses['alignment'] / batch_count:.4f})")


            if best_model_state is not None:
                myModel.load_state_dict(best_model_state)
                log_and_print(f"Loaded best model from epoch {best_epoch}")

            # --- Final Test Section ---
            myModel.eval()
            total_test_acc = 0
            with torch.no_grad():
                eval_target_dataloader = DataLoader(target_dataset, batch_size=BATCH_SIZE, shuffle=False)
                for test_x, test_y, test_ids in eval_target_dataloader:
                    test_x, test_y, test_ids = test_x.to(DEVICE), test_y.to(DEVICE), test_ids.to(DEVICE)
                    outputs, _, _, _ = myModel(test_x, test_ids, 0)
                    test_label, label = G.train_lable2(outputs), G.train_lable2(test_y)
                    total_test_acc += G.acc(test_label, label)

            final_test_acc = (total_test_acc / len(Y_target)) * 100 if len(Y_target) > 0 else 0.0

            log_and_print(
                f"--- Fold {subject_id} training complete, final test accuracy: {final_test_acc:.2f}% (Best epoch: {best_epoch}, Final Beta: {best_beta_history[-1]:.4f}) ---")
            cross_validation_results.append(final_test_acc)

        if cross_validation_results:
            average_accuracy, std_deviation = np.mean(cross_validation_results), np.std(cross_validation_results)
            log_and_print(
                f"\n=================== Final Experiment Results ===================")
            log_and_print(f"Beta Strategy: Progressive Strategy 1 (Linear Growth)")
            log_and_print(f"Beta Parameters: START={BETA_START}, END={BETA_END}, WARMUP_EPOCHS={BETA_WARMUP_EPOCHS}")
            log_and_print(f"All {NUM_SUBJECTS}-fold cross-validation complete!")
            log_and_print(f"Final accuracy for each fold: {[f'{acc:.2f}%' for acc in cross_validation_results]}")
            log_and_print(
                f"Average cross-subject accuracy: {average_accuracy:.2f}% (Standard Deviation: {std_deviation:.2f})")
            log_and_print(
                "===================================================================================================================")

    end_time = time.time()
    total_time = end_time - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = total_time % 60
    print(f"\n总训练时间: {hours:02d}:{minutes:02d}:{seconds:05.2f}")
