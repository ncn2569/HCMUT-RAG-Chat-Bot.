from datetime import datetime
from typing import List, Dict, Optional

_chat_history: List[Dict] = []  # List: [{"user": str, "assistant": str, "timestamp": str}]

def add_turn(user: str, assistant: str, rewritten: Optional[str] = None) -> None:
    """
    Thêm một lượt chat vào history
    """
    turn = {
        "user": user,
        "assistant": assistant,
        "timestamp": datetime.now().isoformat()
    }
    
    if rewritten:
        turn["rewritten"] = rewritten
        
    _chat_history.append(turn)

    # Giữ history trong giới hạn 20 turn
    if len(_chat_history) > 20:
        _chat_history.pop(0)

def get_history() -> List[Dict]:
    """Lấy lịch sử chat"""
    return _chat_history.copy()

def clear_history() -> None:
    """Xóa lịch sử"""
    global _chat_history
    _chat_history = []
    print("History cleared.")

def print_history() -> None:
    """In lịch sử (debug)"""
    print("\n" + "="*60)
    print("CHAT HISTORY")
    print("="*60)
    for i, turn in enumerate(_chat_history, 1):
        print(f"\nTurn {i}:")
        print(f"  User: {turn['user']}")
        if 'rewritten' in turn:
            print(f"  [Rewritten: {turn['rewritten']}]")
        print(f"  Bot: {turn['assistant']}")
    print("="*60)