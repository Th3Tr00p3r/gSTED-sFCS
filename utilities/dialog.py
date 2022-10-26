""" User dialog module. """

from PyQt5.QtWidgets import QMessageBox

YES = QMessageBox.Yes
NO = QMessageBox.No


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
        """Doc."""

        return self._msg_box.exec_()  # warning: blocks!


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

    def __init__(self, txt, title="User Input Needed"):

        super().__init__(msg_icon=QMessageBox.Question, msg_title=title, msg_text=txt)
        self.set_buttons(["Yes", "No"])
        self._msg_box.setDefaultButton(NO)


class NotificationDialog(Dialog):
    """Doc."""

    def __init__(self, txt, title="Notification"):

        super().__init__(msg_icon=QMessageBox.Question, msg_title=title, msg_text=txt)
