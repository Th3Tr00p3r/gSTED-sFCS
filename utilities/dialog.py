# -*- coding: utf-8 -*-
""" User dialog module. """

from PyQt5.QtWidgets import QMessageBox


class UserDialog:
    """Doc."""

    def __init__(
        self,
        msg_icon=QMessageBox.NoIcon,
        msg_title=None,
        msg_text=None,
        msg_inf=None,
        msg_det=None,
    ):
        """Doc."""

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

        return self._msg_box.exec_()


class Error(UserDialog):
    """Doc."""

    def __init__(
        self, type="", msg="", tb="", module="", line="", custom_txt="", custom_title=""
    ):
        """Doc."""

        if type:
            super().__init__(
                msg_icon=QMessageBox.Critical,
                msg_title=f"{type}: {custom_title} ({module}, {line})",
                msg_text=msg,
                msg_det=tb,
            )
        else:
            super().__init__(
                msg_icon=QMessageBox.Critical,
                msg_title=custom_title,
                msg_text=custom_txt,
            )


class Question(UserDialog):
    """Doc."""

    def __init__(self, txt, title="User Input Needed"):
        """Doc."""

        super().__init__(msg_icon=QMessageBox.Question, msg_title=title, msg_text=txt)
        self.set_buttons(["Yes", "No"])
        self._msg_box.setDefaultButton(QMessageBox.No)


class Notification(UserDialog):
    """Doc."""

    def __init__(self, txt, title="Notification"):
        """Doc."""

        super().__init__(msg_icon=QMessageBox.Question, msg_title=title, msg_text=txt)
