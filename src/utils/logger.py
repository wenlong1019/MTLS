import os


def get_filename(log_dir, load):
    if load is None:
        return os.path.join(log_dir, 'train_log.txt')
    else:
        return os.path.join(log_dir, 'test_log.txt')


class Logger(object):
    # Initialize the Logger object with a stream
    def __init__(self, stream):
        # Set the log attribute to None
        self.log = None
        # Set the terminal attribute to the stream
        self.terminal = stream

    # Set the log to a file with the given log_dir and load
    def set_log(self, log_dir, load):
        # Get the filename attribute from the get_filename function with the given log_dir and load
        filename = get_filename(log_dir, load)
        # Set the log attribute to the file
        self.log = open(filename, 'a+')

    # Write a message to both the terminal and the log file
    def write(self, message):
        # Write the message to the terminal
        self.terminal.write(message)
        # Write the message to the log file
        self.log.write(message)

    # Close the log file
    def close(self):
        self.log.close()

    # Flush the terminal
    def flush(self):
        pass
