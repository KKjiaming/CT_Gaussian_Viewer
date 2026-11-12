"""Keep ModernGL/EGL entirely out of the VTK window's event thread and process."""
import multiprocessing as mp
import traceback
import threading


def _worker(connection, parameters, bbox):
    ctx = gpu = None
    try:
        import moderngl
        from gaussian_analysis import AnalysisGPU
        try:
            ctx = moderngl.create_standalone_context(require=330,backend='egl')
        except Exception:
            ctx = moderngl.create_standalone_context(require=330)
        gpu = AnalysisGPU(ctx,parameters,bbox)
        connection.send(('ok',None))
        while True:
            command,args = connection.recv()
            if command=='close':
                break
            try:
                result = getattr(gpu,command)(*args)
                connection.send(('ok',result))
            except Exception:
                connection.send(('error',traceback.format_exc()))
    except (EOFError,BrokenPipeError):
        pass
    except Exception:
        try:
            connection.send(('error',traceback.format_exc()))
        except (EOFError,BrokenPipeError,OSError):
            pass
    finally:
        if gpu is not None:
            gpu.release()
        if ctx is not None:
            ctx.release()
        connection.close()


class AnalysisBackend:
    """Bounded synchronous RPC; VTK remains the only GL owner in the UI process."""
    def __init__(self,parameters,bbox,timeout=15.):
        context = mp.get_context('spawn')
        self.connection,child = context.Pipe()
        self.timeout = timeout
        self.closed = False
        self.lock = threading.RLock()
        self.process = context.Process(target=_worker,args=(child,parameters,bbox),daemon=True)
        self.process.start()
        child.close()
        try:
            self._receive()
        except Exception:
            self.release()
            raise

    def _receive(self):
        if not self.connection.poll(self.timeout):
            self.release()
            raise RuntimeError('GPU analysis timed out; worker stopped to keep the viewer responsive')
        try:
            status,result = self.connection.recv()
        except (EOFError,OSError) as error:
            self.release()
            raise RuntimeError('GPU analysis worker exited unexpectedly') from error
        if status!='ok':
            raise RuntimeError(f'GPU analysis failed:\n{result}')
        return result

    def _call(self,command,*args):
        with self.lock:
            return self._call_locked(command,*args)

    def _call_locked(self,command,*args):
        if self.closed:
            raise RuntimeError('GPU analysis worker is stopped; restart the viewer')
        try:
            self.connection.send((command,args))
        except (BrokenPipeError,EOFError,OSError) as error:
            self.release()
            raise RuntimeError('GPU analysis worker is unavailable; restart the viewer') from error
        return self._receive()

    def slice(self,*args):
        return self._call('slice',*args)

    def projection(self,*args):
        return self._call('projection',*args)

    def update_density(self,*args):
        return self._call('update_density',*args)

    def __enter__(self):
        self.lock.acquire()
        return self

    def __exit__(self,*args):
        self.lock.release()
        return False

    def release(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.connection.send(('close',()))
        except (BrokenPipeError,EOFError,OSError):
            pass
        self.process.join(timeout=.5)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=.5)
        self.connection.close()
