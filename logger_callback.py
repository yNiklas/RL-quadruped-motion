from stable_baselines3.common.callbacks import BaseCallback

class LoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self):
        infos = self.locals.get("infos", [])
        for info in infos:
            if "collapse/z_bound" in info:
                self.logger.record("collapse/z_bound", info["collapse/z_bound"])
            if "collapse/roll_bound" in info:
                self.logger.record("collapse/roll_bound", info["collapse/roll_bound"])
            if "collapse/pitch_bound" in info:
                self.logger.record("collapse/pitch_bound", info["collapse/pitch_bound"])

        keys = [
            "state/body_v",
            "state/body_z",
            "state/roll",
            "state/pitch",
            "reward/r_vx",
            "reward/r_z",
            "reward/r_homing_similarity",
            "reward/r_action_similarity",
            "reward/r_vz",
            "reward/r_orientation",
            "reward/r_upright",
            "reward/r_zmp"
        ]

        # Compute mean across parallel environments
        values = {k: [] for k in keys}
        for info in infos:
            for k in keys:
                if k in info:
                    values[k].append(info[k])
        for k in keys:
            if values[k]:
                mean_val = sum(values[k]) / len(values[k])
                self.logger.record(k, mean_val)
        return True
