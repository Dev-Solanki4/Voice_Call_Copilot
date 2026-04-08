class MatchIntentResult:
  def __init__(self, success: bool, score: float):
    self.success = success
    self.score = score

class SetGuidelinesResult:
  def __init__(self, success: bool, message: str):
    self.success = success
    self.message = message