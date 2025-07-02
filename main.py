import os
import pandas as pd
from configparser import ConfigParser
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from handler import FuzzyIndexDataset, FuzzyStreamer
from helper import AverageMeter
from model import FDDR

# Carregar configurações
cfg = ConfigParser()
cfg.read('./config.ini')

# Obter parâmetros principais
data_src = cfg.get('default', 'data_src')
log_src = cfg.get('default', 'log_src')
epochs = cfg.getint('default', 'epochs')
save_per_epoch = cfg.getint('default', 'save_per_epoch')
c = cfg.getfloat('default', 'c')
lag = cfg.getint('default', 'lag')
fuzzy_degree = cfg.getint('fddrl', 'fuzzy_degree')

# Criar diretórios necessários
os.makedirs(os.path.join(data_src, 'futures', 'train'), exist_ok=True)
os.makedirs(os.path.join(data_src, 'futures', 'test'), exist_ok=True)
os.makedirs(os.path.join(data_src, 'fuzzy_futures', 'train'), exist_ok=True)
os.makedirs(os.path.join(data_src, 'fuzzy_futures', 'test'), exist_ok=True)
os.makedirs(log_src, exist_ok=True)

#################### Data Preparation ####################
print("Preparando dados...")

# Carregar e preparar dados
df = pd.read_csv('eurusdt_5min.csv')
df['CloseDiff'] = df['close'].diff()
df = df.dropna()

# Dividir em treino e teste (80/20)
train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False)

# Salvar dados de treino e teste
train_df.to_csv(os.path.join(data_src, 'futures', 'train', 'train_data.csv'), index=False)
test_df.to_csv(os.path.join(data_src, 'futures', 'test', 'test_data.csv'), index=False)

print("Dados brutos preparados e salvos.")

#################### Fuzzy Preprocessing ####################
print("Verificando pré-processamento fuzzy...")

train_fuzzy_dir = os.path.join(data_src, 'fuzzy_futures', 'train')
test_fuzzy_dir = os.path.join(data_src, 'fuzzy_futures', 'test')

# Verificar e executar pré-processamento para treino
if len(os.listdir(train_fuzzy_dir)) == 0:
    print("Executando pré-processamento para dados de treino...")
    streamer = FuzzyStreamer(lag, fuzzy_degree)
    streamer.transform(os.path.join(data_src, 'futures', 'train'), train_fuzzy_dir)
    print(f"Pré-processamento de treino completo! Arquivos gerados: {len(os.listdir(train_fuzzy_dir))}")
else:
    print("Dados fuzzy de treino já existem. Pulando pré-processamento.")

# Verificar e executar pré-processamento para teste
if len(os.listdir(test_fuzzy_dir)) == 0:
    print("Executando pré-processamento para dados de teste...")
    streamer = FuzzyStreamer(lag, fuzzy_degree)
    streamer.transform(os.path.join(data_src, 'futures', 'test'), test_fuzzy_dir)
    print(f"Pré-processamento de teste completo! Arquivos gerados: {len(os.listdir(test_fuzzy_dir))}")
else:
    print("Dados fuzzy de teste já existem. Pulando pré-processamento.")

############################################################

# Carregar datasets
print("Carregando datasets...")
train_dataset = FuzzyIndexDataset(train_fuzzy_dir, lag)
test_dataset = FuzzyIndexDataset(test_fuzzy_dir, lag)

train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=1)
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=1)

print(f"Tamanho do train_dataloader: {len(train_dataloader)}")
print(f"Tamanho do test_dataloader: {len(test_dataloader)}")

if len(train_dataloader) == 0 or len(test_dataloader) == 0:
    print("ERRO: Dataloaders vazios! Verifique:")
    print(f"1. Diretório de treino fuzzy: {train_fuzzy_dir}")
    print(f"2. Diretório de teste fuzzy: {test_fuzzy_dir}")
    print(f"3. Número de arquivos em treino: {len(os.listdir(train_fuzzy_dir))}")
    print(f"4. Número de arquivos em teste: {len(os.listdir(test_fuzzy_dir))}")
    exit(1)
    
# Modelo e otimizador
print("Inicializando modelo...")
fddr = FDDR(lag, fuzzy_degree)
optimizer = torch.optim.Adam(fddr.parameters())

# Medidores de desempenho
train_reward_meter = AverageMeter(epochs, len(train_dataloader))
test_reward_meter = AverageMeter(epochs, len(test_dataloader))

# Fase de treinamento
print("Iniciando treinamento...")
for e in range(epochs):
    with tqdm(total=len(train_dataloader), ncols=130) as progress_bar:
        # Modo de treino
        fddr.train()
        for i, (returns, fragments, mean, var) in enumerate(train_dataloader):
            # Computar ações usando FDDR
            delta = fddr(fragments, running_mean=mean, running_var=var).double().squeeze(-1)

            # Computar recompensa
            pad_delta = F.pad(delta, [1, 0])
            delta_diff = (pad_delta[:, 1:] - pad_delta[:, :-1])
            reward = torch.sum(delta * returns - c * torch.abs(delta_diff))

            # Atualizar FDDR
            optimizer.zero_grad()
            (-reward).backward()
            optimizer.step()

            # Registrar e mostrar informações
            train_reward_meter.append(reward.item())
            progress_bar.set_description(
                '[Epoch %d/%d][Batch %d/%d][Reward: %.4f]' %
                (e+1, epochs, i+1, len(train_dataloader), train_reward_meter.get_average(-1)))
            progress_bar.update()

        # Modo de avaliação
        fddr.eval()
        with torch.no_grad():
            for i, (returns, fragments, mean, var) in enumerate(test_dataloader):
                # Computar ações usando FDDR
                delta = fddr(fragments, running_mean=mean, running_var=var).double().squeeze(-1)

                # Computar recompensa
                pad_delta = F.pad(delta, [1, 0])
                delta_diff = (pad_delta[:, 1:] - pad_delta[:, :-1])
                reward = torch.sum(delta * returns - c * torch.abs(delta_diff))

                test_reward_meter.append(reward.item())

        # Atualizar barra de progresso com métricas finais da época
        progress_bar.set_description(
            '[Epoch %d/%d][Train: %.4f][Test: %.4f]' %
            (e+1, epochs, train_reward_meter.get_average(-1), test_reward_meter.get_average(-1)))
        progress_bar.refresh()

        # Salvar modelo periodicamente
        if (e % save_per_epoch == 0) or (e == epochs - 1):
            torch.save(fddr.state_dict(), os.path.join(log_src, f'fddrl_epoch_{e}.pkl'))
        
        # Atualizar medidores
        train_reward_meter.step()
        test_reward_meter.step()

# Salvar modelo e histórico final
print("Treinamento completo! Salvando resultados finais...")
torch.save(fddr.state_dict(), os.path.join(log_src, 'fddrl_final.pkl'))
np.save(os.path.join(log_src, 'fddrl_train_reward.npy'), train_reward_meter.get_average())
np.save(os.path.join(log_src, 'fddrl_test_reward.npy'), test_reward_meter.get_average())

# Plotar curva de recompensa
plt.figure(figsize=(12, 6))
plt.plot(train_reward_meter.get_average(), label='Treino')
plt.plot(test_reward_meter.get_average(), label='Teste')
plt.title('Recompensa por Época')
plt.xlabel('Época')
plt.ylabel('Recompensa Média')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(log_src, 'reward_curve.png'))
plt.show()

print("Processo concluído com sucesso!")