class AIController:
    def __init__(self):
        self.ultimo_pulo = 0
        self.pulos_excessivos = 0
        self.min_frames_entre_pulos = 20
        self.distancia_obstaculo_segura = 200
        
    def processar_acao(self, env, model, obs, frame_count):
        action, _states = model.predict(obs, deterministic=True)
        
        if action == 2:  # Ação de pulo
            if frame_count - self.ultimo_pulo < self.min_frames_entre_pulos:
                obstaculo_proximo = self._verifica_obstaculo_proximo(env)
                
                if not obstaculo_proximo:
                    action = 0  # Cancelar pulo
                    self.pulos_excessivos += 1
            else:
                self.ultimo_pulo = frame_count
                
        return action
    
    def _verifica_obstaculo_proximo(self, env):
        if hasattr(env, 'obstacles'):
            for obstacle in env.obstacles:
                dist_x = obstacle['x'] - env.agent_x
                if 0 < dist_x < self.distancia_obstaculo_segura:
                    return True
        return False
