# @Time : 2022/10/25 13:47
# @Author : CaoXiang
# @配置相关
from abc import ABCMeta, abstractmethod
import yaml
from util.exception import ParamLoadError


class BaseConfigLoader(metaclass=ABCMeta):
    """
        配置信息加载基类，考虑两种加载模式，优先从配置文件加载，然后再考虑从argparse加载。
    """
    opt_dict = {}
    config_dict = {}

    @abstractmethod
    def _load_config(self):
        pass

    def attempt_load_param(self, param_name:str):
        if self.opt_dict and self.opt_dict.get(param_name):
            return self.opt_dict[param_name]
        elif param_name in self.config_dict.keys():
            return self.config_dict[param_name]
        else:
            raise ParamLoadError(param_name)


class YamlConfigLoader(BaseConfigLoader):
    def __init__(self, yaml_path, opt_dict=None):
        self.yaml_path = yaml_path
        self.config_dict = {}
        self._load_config()
        self.opt_dict = opt_dict if opt_dict is not None else {}

    def _load_config(self):
        with open(self.yaml_path) as f:
            self.config_dict = yaml.load(f, Loader=yaml.FullLoader)


class DictConfigLoader(BaseConfigLoader):
    def __init__(self, config_dict:dict, opt_dict=None):
        self.config_dict = config_dict
        self.opt_dict = opt_dict if opt_dict is not None else {}

    def _load_config(self):
        pass


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default=None, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt_dict = vars(parser.parse_args())
    yaml_path = "config.yaml"
    yaml_loader = YamlConfigLoader(yaml_path, opt_dict)
    print(yaml_loader.attempt_load_param("device"))
    config_dict = {"device": 3}
    dict_loader = DictConfigLoader(config_dict, opt_dict)
    print(dict_loader.attempt_load_param("device"))
