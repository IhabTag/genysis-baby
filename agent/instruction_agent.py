from typing import List, Dict, Any
from agent.command_parser import CommandParser
from agent.text_actions import TextActionAgent


class InstructionAgent:
    """
    Step 16:
    High-level instruction-following agent.

    Usage:
        ia = InstructionAgent(curiosity_agent, text_action_agent, env)
        ia.execute("click search and type 'hello world'")
    """

    def __init__(self, curious_agent, text_action_agent: TextActionAgent, env):
        self.curious = curious_agent
        self.text_actions = text_action_agent
        self.env = env

        self.parser = CommandParser()

    # -----------------------------------------------------------
    # Execute natural language instruction
    # -----------------------------------------------------------
    def execute(self, instruction: str) -> List[Dict[str, Any]]:
        """
        Returns:
          list of results per step:
            [
              {"op":"click","success":True},
              {"op":"type","success":True},
            ]
        """
        plan = self.parser.parse(instruction)
        results = []

        obs = self.env._observe()  # current frame

        for step in plan:
            op = step["op"]

            if op == "click":
                success = self.text_actions.find_and_click(obs["image"], step["text"])
                results.append({"op": "click", "target": step["text"], "success": success})

            elif op == "type":
                success = self.text_actions.type_into(
                    obs["image"],
                    target_text=step["text"],
                    content=step["content"],
                )
                results.append({"op": "type", "target": step["text"], "content": step["content"], "success": success})

            elif op == "scroll":
                success = self._apply_scroll(step["amount"])
                results.append({"op": "scroll", "amount": step["amount"], "success": success})

            else:
                results.append({"op": op, "success": False})

            # update obs for next step
            obs = self.env._observe()

        return results

    # -----------------------------------------------------------
    # Internal: apply scroll via environment executor
    # -----------------------------------------------------------
    def _apply_scroll(self, amount: int) -> bool:
        try:
            self.env.executor.execute({"type": "SCROLL", "amount": amount})
            return True
        except:
            return False
