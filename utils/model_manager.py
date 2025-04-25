import os
from stable_baselines3 import DQN
from env.jogo_com_obstaculos import JogoComObstaculos
import numpy as np

class ModelManager:
    def __init__(self, projeto_dir):
        self.projeto_dir = projeto_dir
        self.model_path = os.path.join(projeto_dir, "models/dqn_jogo_obstaculos.zip")
        self.model_path_v2 = os.path.join(projeto_dir, "models/dqn_jogo_obstaculos_v2.zip")
        self.experiencias_path = os.path.join(projeto_dir, "models/experiencias.npz")
        self.experiencias_buffer = []
        
    def treinar_modelo(self):
        print("Treinando novo modelo...")
        env = JogoComObstaculos(render_mode=None)
        model = DQN(
            "MlpPolicy", 
            env, 
            verbose=1,
            learning_rate=0.0005,
            buffer_size=50000,
            exploration_fraction=0.2,
            exploration_final_eps=0.05
        )
        model.learn(total_timesteps=25000)
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        model.save(self.model_path)
        env.close()
        return model
        def treinar_modelo_incremental(self, experiencias, timesteps=5000):
            print(f"Treinando modelo incrementalmente com {len(experiencias)} experiências...")
            env = JogoComObstaculos(render_mode=None)
            
            if os.path.exists(self.model_path_v2):
                model = DQN.load(self.model_path_v2)
                modelo_base = self.model_path_v2
            elif os.path.exists(self.model_path):
                model = DQN.load(self.model_path)
                modelo_base = self.model_path
            else:
                model = self.treinar_modelo()
                modelo_base = self.model_path
                
            model.set_env(env)
            
            print("Adicionando experiências ao buffer de replay...")
            for exp in experiencias:
                obs, action, reward, next_obs, done = exp
                model.replay_buffer.add(obs, next_obs, action, reward, done, [{}])
                
            model.learn(total_timesteps=timesteps)
            
            os.makedirs(os.path.dirname(self.model_path_v2), exist_ok=True)
            model.save(self.model_path_v2)
            print(f"Modelo incrementado salvo em {self.model_path_v2} (baseado em {modelo_base})")
            
            env.close()
            return model
        
        def adicionar_experiencia(self, obs, action, reward, next_obs, done):
            self.experiencias_buffer.append((obs, action, reward, next_obs, done))
        
        def salvar_experiencias(self):
            if not self.experiencias_buffer:
                print("Sem experiências para salvar.")
                return
                
            print(f"Salvando {len(self.experiencias_buffer)} experiências...")
            
            experiencias_existentes = []
            if os.path.exists(self.experiencias_path):
                with np.load(self.experiencias_path, allow_pickle=True) as dados:
                    experiencias_existentes = dados['experiencias'].tolist()
                
            todas_experiencias = experiencias_existentes + self.experiencias_buffer
            
            max_experiencias = 50000
            if len(todas_experiencias) > max_experiencias:
                todas_experiencias = todas_experiencias[-max_experiencias:]
                
            os.makedirs(os.path.dirname(self.experiencias_path), exist_ok=True)
            np.savez_compressed(
                self.experiencias_path, 
                experiencias=np.array(todas_experiencias, dtype=object)
            )
            print(f"Experiências salvas em {self.experiencias_path}")
            
            self.experiencias_buffer = []
        
        def carregar_experiencias(self):
            if os.path.exists(self.experiencias_path):
                print(f"Carregando experiências de {self.experiencias_path}")
                with np.load(self.experiencias_path, allow_pickle=True) as dados:
                    return dados['experiencias'].tolist()
            return []

        def carregar_modelo(self):
            if os.path.exists(self.model_path_v2):
                print(f"Carregando modelo avançado de {self.model_path_v2}")
                return DQN.load(self.model_path_v2)
            elif os.path.exists(self.model_path):
                print(f"Carregando modelo básico de {self.model_path}")
                return DQN.load(self.model_path)
            else:
                print("Nenhum modelo encontrado. Treinando novo modelo...")
                return self.treinar_modelo()
