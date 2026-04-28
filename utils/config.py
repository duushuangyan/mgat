"""
配置管理模块
"""

import collections.abc
import logging
import os
import yaml
from pathlib import Path
from typing import Any, Optional, Union, List


class dotdict(dict):
    """
    支持点符号访问的字典类
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取字典中键对应的值，如果键不存在则返回默认值
        """
        return dict.get(self, key, default)

    def recursive_update(self, u: dict) -> 'dotdict':
        """
        递归更新字典，支持嵌套字典的深度合并
        """
        return dotdict._recursive_update(self, u)

    @staticmethod
    def _recursive_update(d: dict, u: dict) -> dict:
        """递归更新的静态方法实现"""
        for k, v in u.items():
            if isinstance(v, collections.abc.Mapping):
                d[k] = dotdict._recursive_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    def to_dict(self) -> dict:
        """
        将dotdict递归转换回普通字典
        """
        result = {}
        for key, value in self.items():
            if isinstance(value, dotdict):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                result[key] = [
                    item.to_dict() if isinstance(item, dotdict) else item
                    for item in value
                ]
            else:
                result[key] = value
        return result


def _convert_to_dotdict(regular_dict: Any) -> Any:
    """
    将普通字典递归转换为dotdict
    """
    if not isinstance(regular_dict, dict):
        return regular_dict

    for key in regular_dict:
        if isinstance(regular_dict[key], dict):
            regular_dict[key] = _convert_to_dotdict(regular_dict[key])
        elif isinstance(regular_dict[key], list):
            # 处理列表中的每个元素，如果是字典则转换
            regular_dict[key] = [
                _convert_to_dotdict(item) if isinstance(item, dict) else item
                for item in regular_dict[key]
            ]
    return dotdict(regular_dict)


def load_config(config_path: Union[str, Path]) -> dotdict:
    """
    加载YAML配置文件并转换为支持点符号访问的dotdict对象
    """
    config_path = Path(config_path) if isinstance(config_path, str) else config_path

    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        try:
            params = yaml.safe_load(f)
            if params is None:
                return dotdict({})
            return _convert_to_dotdict(params)
        except yaml.YAMLError as exc:
            raise yaml.YAMLError(f"YAML解析错误 ({config_path}): {exc}")


def save_config(config: Union[dict, dotdict], save_path: Union[str, Path]) -> None:
    """
    保存配置到YAML文件
    """
    save_path = Path(save_path) if isinstance(save_path, str) else save_path

    # 确保父目录存在
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # 如果是dotdict，转换为普通字典
    if isinstance(config, dotdict):
        config = config.to_dict()

    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)


def merge_configs(*config_paths: Union[str, Path]) -> dotdict:
    """
    合并多个配置文件
    """
    merged = dotdict({})
    for path in config_paths:
        config = load_config(path)
        merged.recursive_update(config)
    return merged


def load_all_configs(config_dir: Union[str, Path] = 'config') -> dotdict:
    """
    加载指定目录下的所有配置文件并合并
    """
    config_dir = Path(config_dir) if isinstance(config_dir, str) else config_dir

    config_files = {
        'data': 'data_config.yaml',
        'preprocess': 'preproces_config.yaml',
        'model': 'model_config.yaml',
        'train': 'train_config.yaml'
    }

    all_configs = dotdict({})

    for key, filename in config_files.items():
        filepath = config_dir / filename
        if filepath.exists():
            all_configs[key] = load_config(filepath)
        else:
            logging.warning(f"配置文件不存在，跳过: {filepath}")
            all_configs[key] = dotdict({})

    return all_configs


def setup_logging(
        save_dir: Union[str, Path],
        log_level: str = 'INFO',
        log_name: str = 'MGAT-Training'
) -> logging.Logger:
    """
    设置日志系统
    """
    save_dir = Path(save_dir) if isinstance(save_dir, str) else save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    log_file = save_dir / 'train.log'

    # 获取或创建日志器
    logger = logging.getLogger(log_name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # 清除已有的处理器（避免重复添加）
    logger.handlers.clear()

    # 配置日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(getattr(logging, log_level.upper()))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


# 为了向后兼容，保留原有的函数名别名
load_yaml_into_dotdict = load_config
convert_to_dot_dict = _convert_to_dotdict