# utils2/hud.py
import pybullet as p

class DebugHUD:
    def __init__(self, size=1.2, z=0.05, color=(1,1,1)):
        self.size  = float(size)
        self.z     = float(z)
        self.color = color
        self._ids  = []   # stable list of item ids
        self._last_anchor = None

    def follow(self, anchor_xy, lines):
        x, y = float(anchor_xy[0]), float(anchor_xy[1])
        # ensure stable length of ids
        while len(self._ids) < len(lines):
            self._ids.append(-1)

        # update/replace each line
        for i, text in enumerate(lines):
            pos = [x, y, self.z + 0.05*i]
            uid_prev = self._ids[i] if i < len(self._ids) else -1
            uid = p.addUserDebugText(
                text=text,
                textPosition=pos,
                textColorRGB=self.color,
                textSize=self.size,
                lifeTime=0,                 # persist, but REPLACED below
                replaceItemUniqueId=uid_prev
            )
            self._ids[i] = uid

        # if fewer lines than before: remove the extras
        if len(lines) < len(self._ids):
            for j in range(len(lines), len(self._ids)):
                if self._ids[j] >= 0:
                    try: p.removeUserDebugItem(self._ids[j])
                    except: pass
            del self._ids[len(lines):]
