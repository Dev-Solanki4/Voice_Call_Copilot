from typing import Optional

class CharacterVision:
  def __init__(self, success: bool, message: Optional[str] = None, image: Optional[str] = None):
    self.image = image
    self.success = success
    self.message = message