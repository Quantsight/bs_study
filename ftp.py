import ftplib
import yaml

def put(path, fn, config_fn):
    config = yaml.safe_load(open(config_fn))
    ftp = ftplib.FTP(config['host'])
    ftp.login(config['user'], config['pw'])
    ftp.cwd(config['path_root'] + path)
    ftp.storbinary("STOR " + fn, open(fn, "rb"))