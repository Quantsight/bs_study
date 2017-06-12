import ftplib

def put(in_fn, out_path, out_fn, config):
    ftp = ftplib.FTP(config['host'])
    ftp.login(config['user'], config['pw'])
    ftp.cwd(config['path_root'] + out_path)
    with open(in_fn, "rb") as f:
        ftp.storbinary("STOR " + out_fn, f)