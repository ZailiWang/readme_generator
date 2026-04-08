import os
from typing import Dict

def backup_proxy_in_process(proxy_backup: Dict) -> Dict:
    proxy_vars = [
        'HTTP_PROXY', 'HTTPS_PROXY', 'FTP_PROXY', 'SOCKS_PROXY',
        'http_proxy', 'https_proxy', 'ftp_proxy', 'socks_proxy',
        'NO_PROXY', 'no_proxy'
    ]
    proxy_backup.update({var: os.environ[var] for var in proxy_vars if var in os.environ})
    print(f"已备份代理环境变量：{list(proxy_backup.keys())}")
    return proxy_backup

def clear_proxy_in_process(proxy_backup: Dict) -> Dict:
    proxy_backup = backup_proxy_in_process(proxy_backup)
    proxy_vars = list(proxy_backup.keys())
    for var in proxy_vars:
        if var in os.environ:
            del os.environ[var]
            print(f"已清除环境变量：{var}")
    return proxy_backup

def restore_proxy_in_process(proxy_backup: Dict):
    if not proxy_backup:
        print("没有备份的代理配置，无需恢复")
        return
    for var, value in proxy_backup.items():
        os.environ[var] = value
        print(f"已恢复环境变量: {var}={value}")
    return proxy_backup