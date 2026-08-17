# Original COMMAG PPO baseline

The upstream COMMAG repository publishes the original PPO agents and encoder used in the accompanying experimental work. Those artifacts depend on a historical TensorFlow/stable-baselines environment that is not silently converted by this repository.

The publication benchmark therefore treats the original PPO result as an explicit compatibility baseline:

1. run the upstream models in their compatible environment/container;
2. export one row per evaluated transition with `episode_id,selected_action`;
3. place the file at `data/prepared/commag-publication/original_ppo_actions.csv.gz`;
4. rerun the publication evaluation so the exact upstream decisions are scored with the same direct-method critics and paired episode statistics.

This avoids claiming that a reimplemented policy is the "original PPO" when the published weights could not be loaded exactly.
