import os
from stable_baselines3 import DQN
from env.jogo_com_obstaculos import JogoComObstaculos

class ModelManager:
    def __init__(self, projeto_dir):
        self.projeto_dir = projeto_dir
        self.model_path = os.path.join(projeto_dir, "models/dqn_jogo_obstaculos.zip")
        self.model_path_v2 = os.path.join(projeto_dir, "models/dqn_jogo_obstaculos_v2.zip")
        
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
