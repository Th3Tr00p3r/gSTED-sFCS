# -*- coding: utf-8 -*-
""" User dialog module. """

import os
import traceback
import sys
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

    def __init__(self, exc=None, error_txt="", error_title=""):
        """Doc."""

        if exc:
            self.exc_type, _, self.tb = sys.exc_info()
            self.error_type = self.exc_type.__name__
            self.fname = os.path.split(self.tb.tb_frame.f_code.co_filename)[1]
            self.lineno = self.tb.tb_lineno
            super().__init__(
                msg_icon=QMessageBox.Critical,
                msg_title=self.error_type + ": " + error_txt,
                msg_text=str(exc),
                msg_inf=("module: {}" + "\n" + "line: {}").format(
                    self.fname, self.lineno
                ),
                msg_det="\n".join(traceback.format_tb(self.tb)),
            )
        else:
            super().__init__(
                msg_icon=QMessageBox.Critical,
                msg_title=error_title,
                msg_text=error_txt,
            )


class Question(UserDialog):
    """Doc."""

    def __init__(self, q_txt, q_title="User Input Needed"):
        """Doc."""

        super().__init__(
            msg_icon=QMessageBox.Question, msg_title=q_title, msg_text=q_txt
        )
        self.set_buttons(["Yes", "No"])
        self._msg_box.setDefaultButton(QMessageBox.No)
