import dotenv
import hydra
from omegaconf import DictConfig
import sys

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig):

    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934
    from src.train import train
    from src.utils import utils

    # A couple of optional utilities:
    # - disabling python warnings
    # - easier access to debug mode
    # - forcing debug friendly configuration
    # - forcing multi-gpu friendly configuration
    # You can safely get rid of this line if you don't want those
    utils.extras(config)  # 增加上述设定
    utils.save_config(config)  # 保存config到指定位置

    # Pretty print config using Rich library
    # 以标准yaml形式打印config
    if config.get("print_config"):
        utils.print_config(config, resolve=True)

    # Train model
    return train(config)


if __name__ == "__main__":
    # 避免输出大量无效信息
    # sys.tracebacklimit = 0
    main()
