"""Training and evaluation"""

import run_lib
from absl import app, flags
from ml_collections.config_flags import config_flags
import os


FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    'config', 'configs/inference_ribodiffusion.py', 'Training configuration.', lock_config=True
)
flags.DEFINE_enum('mode', 'inference', ['inference'],
                  'Running mode')
flags.DEFINE_string('save_folder', 'exp_inf', 'The folder name for storing inference results')
flags.DEFINE_string('PDB_file', 'example/R1107.pdb', 'The PDB file for inference')
flags.DEFINE_boolean('deterministic', True, 'Set random seed for reproducibility.')
flags.mark_flags_as_required(['PDB_file'])


def main(argv):
    # Set random seed
    if FLAGS.deterministic:
        run_lib.set_random_seed(FLAGS.config)

    if FLAGS.mode == 'inference':
        run_lib.vpsde_inference(FLAGS.config, FLAGS.save_folder, FLAGS.PDB_file)
    else:
        raise ValueError(f"Mode {FLAGS.mode} not recognized.")


if __name__ == '__main__':
    app.run(main)
