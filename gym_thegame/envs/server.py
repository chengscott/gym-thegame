import subprocess


class Server:
  def __init__(self, server='./thegame-server', port=50051):
    self.cmd = [server, '-listen', ':{}'.format(port)]

  def start(self):
    self.proc = subprocess.Popen(
        self.cmd, stdin=subprocess.PIPE, encoding='utf-8')

  def _send(self, cmd):
    self.proc.stdin.write(cmd)
    self.proc.stdin.flush()

  def pause(self):
    self._send('p\n')

  def resume(self):
    self._send('r\n')

  def sync(self):
    self._send('s\n')

  def terminate(self):
    self.proc.kill()
