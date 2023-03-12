""" User dialog module. """

from PyQt5.QtWidgets import QMessageBox

input_dict = {
    QMessageBox.Yes: True,
    QMessageBox.No: False,
    QMessageBox.Cancel: None,
    QMessageBox.Ok: None,
}


class Dialog:
    """Doc."""

    def __init__(
        self,
        msg_icon=QMessageBox.NoIcon,
        msg_title=None,
        msg_text=None,
        msg_inf=None,
        msg_det=None,
    ):

        self._msg_box = QMessageBox()
        self._msg_box.setIcon(msg_icon)
        self._msg_box.setWindowTitle(msg_title)
        self._msg_box.setText(msg_text)
        self._msg_box.setInformativeText(msg_inf)
        self._msg_box.setDetailedText(msg_det)

    def set_buttons(self, std_bttn_list):
        """Doc."""

        for std_bttn in std_bttn_list:
            self._msg_box.addButton(getattr(QMessageBox, std_bttn))

    def display(self):
        """Show dialog to user, wait for response and translate it into boolean (or None in case of pressing cancel)."""

        user_input = self._msg_box.exec_()  # warning: blocks!
        return input_dict[user_input]


class ErrorDialog(Dialog):
    """Doc."""

    def __init__(self, type="", loc="", msg="", tb="", custom_txt="", custom_title=""):

        if type:
            location_string = " -> ".join([f"{filename}, {lineno}" for filename, lineno in loc[:2]])
            super().__init__(
                msg_icon=QMessageBox.Critical,
                msg_title=f"{type}: {custom_title} ({location_string})",
                msg_text=msg,
                msg_det=tb,
            )
        else:
            super().__init__(
                msg_icon=QMessageBox.Critical,
                msg_title=custom_title,
                msg_text=custom_txt,
            )


class QuestionDialog(Dialog):
    """Doc."""

    def __init__(self, txt, title="User Input Needed", should_include_cancel=False):

        super().__init__(msg_icon=QMessageBox.Question, msg_title=title, msg_text=txt)
        if should_include_cancel:
            self.set_buttons(["Yes", "No", "Cancel"])
            self._msg_box.setDefaultButton(QMessageBox.Cancel)
        else:
            self.set_buttons(["Yes", "No"])
            self._msg_box.setDefaultButton(QMessageBox.No)


class NotificationDialog(Dialog):
    """Doc."""

    def __init__(self, txt, title="Notification"):

        super().__init__(msg_icon=QMessageBox.Question, msg_title=title, msg_text=txt)
