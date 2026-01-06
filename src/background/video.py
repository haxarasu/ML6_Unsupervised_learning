from __future__ import annotations
import cv2
import numpy as np


class Video:
    def __init__(self, path: str) -> None:
        self.path = path
        self.frames: list[np.ndarray] = []
        self.frame_shape = None
        self.X = None
        self.U = None
        self.s = None
        self.Vt = None
        self.to_matrix_called = False


    def first_frame_raw(self) -> np.ndarray:
        cap = cv2.VideoCapture(self.path)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"Cannot read first frame from: {self.path}")

        return frame


    def read(self) -> Video:
        cap = cv2.VideoCapture(self.path)

        self.frames = []

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            frame = frame.astype(np.float32)
            frame /= 255.0

            self.frames.append(frame)

        cap.release()

        if not self.frames:
            raise RuntimeError(f"No frames read from: {self.path}")

        self.frame_shape = tuple(self.frames[0].shape)
        self.X = self.U = self.s = self.Vt = None
        self.to_matrix_called = False

        return self


    def to_matrix(self) -> np.ndarray:
        if not self.frames:
            raise RuntimeError("Call read() before calling to_matrix()")
            
        # X shape: (pixels, frames)
        self.X = np.stack([f.reshape(-1) for f in self.frames], axis=1).astype(np.float32, copy=False)
        self.to_matrix_called = True

        return self.X


    def compute_svd(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.to_matrix_called is False:
            raise RuntimeError("Call to_matrix() before calling compute_svd()")

        # full SVD
        U, s, Vt = np.linalg.svd(self.X, full_matrices=False)
        self.U, self.s, self.Vt = U, s, Vt

        return U, s, Vt


    def reconstruct_first_frame(self, *, rank: int = 1, clip: bool = True) -> np.ndarray:
        if self.U is None or self.s is None or self.Vt is None:
            raise RuntimeError("Call compute_svd() before calling reconstruct_first_frame()")
        
        r = max(1, int(rank))

        U_r = self.U[:, :r]
        s_r = self.s[:r]
        Vt_r = self.Vt[:r, :]

        x0 = (U_r * s_r) @ Vt_r[:, 0]
        img = x0.reshape(self.frame_shape)

        if clip:
            img = np.clip(img, 0.0, 1.0)

        return img