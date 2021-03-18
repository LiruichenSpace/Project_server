import threading
import time

class threaed_handler(threading.Thread):
    def __init__(self, thread_list, logger):
        super().__init__()
        self.thread_list=thread_list
        self.logger=logger

    def run(self) -> None:
        while True:
            for i in range(len(self.thread_list) - 1, -1, -1):  # 倒序遍历
                if self.thread_list[i].is_terminated:
                    self.thread_list.pop(i)
            time.sleep(30)# clean disconnected thread exery 30 sec