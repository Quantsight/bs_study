import ftplib
import yaml

def put(in_fn, out_path, out_fn, config_fn):
    ftp_config = yaml.safe_load(open(config_fn))['ftp']
    ftp = ftplib.FTP(ftp_config['host'])
    ftp.login(ftp_config['user'], ftp_config['pw'])
    ftp.cwd(ftp_config['path_root'] + out_path)
    with open(in_fn, "rb") as f:
        ftp.storbinary("STOR " + out_fn, f)