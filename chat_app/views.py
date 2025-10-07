from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt # For simplicity in this example, consider CSRF protection for production
import json

# Attempt to import your agent logic. This file needs to be created by you.
# It should contain your LangChain setup (LLM, DB, tools, agent, chains, etc.)
# and the SQLApprovalRequired exception.
try:
    from . import agent_logic
except ImportError:
    # Provide a dummy agent_logic if the real one isn't set up yet, so the app can run.
    class DummySQLApprovalRequired(Exception):
        def __init__(self, sql, tool_input):
            self.sql = sql
            self.tool_input = tool_input
            super().__init__("SQL Approval Required (Dummy)")

    class agent_logic:
        SQLApprovalRequired = DummySQLApprovalRequired

        @staticmethod
        def initialize_agent():
            print("Dummy agent initialized. Implement agent_logic.py")
            # In your real agent_logic.py, this would initialize your actual agent_executor
            pass

        @staticmethod
        def run_agent_invoke(prompt, chat_history, agent_scratchpad):
            print(f"Dummy agent received prompt: {prompt}")
            # Simulate agent behavior
            if "customer" in prompt.lower():
                 raise agent_logic.SQLApprovalRequired("SELECT COUNT(*) FROM customers;", prompt)
            return {"output": f"This is a dummy response to: {prompt}"}

        @staticmethod
        def run_final_chain(question, sql, result):
            return f"Dummy final answer for Q: {question} SQL: {sql} Result: {result}"
        
        @staticmethod
        def execute_sql_query(sql_query):
            print(f"Dummy executing SQL: {sql_query}")
            return f"[(91,)] # Dummy result for {sql_query}"

# Initialize your agent components (this might be done once when Django starts)
# For simplicity, we call it here, but consider Django's app loading mechanisms for production.
agent_logic.initialize_agent()


def _clear_pending_sql_state(request):
    """Remove any session keys that track outstanding SQL approvals."""
    keys_to_clear = [
        'pending_sql',
        'pending_question',
        'pending_tool_input',
        'pending_agent_scratchpad',
        'resume_chat_history',
    ]
    for key in keys_to_clear:
        if key in request.session:
            del request.session[key]


def _execute_sql_and_resume(
    request,
    sql_to_execute,
    original_question,
    interrupted_scratchpad,
    current_chat_history_for_resume,
    chat_history,
):
    """Execute SQL, feed the results back into the agent, and return the response."""

    sql_execution_output = agent_logic.execute_sql_query(sql_to_execute)

    if isinstance(sql_execution_output, str):
        chat_history.append({"role": "assistant", "content": sql_execution_output})
        request.session['chat_history'] = chat_history
        _clear_pending_sql_state(request)
        return JsonResponse({'error': sql_execution_output})
    elif isinstance(sql_execution_output, dict):
        sql_result_json_str = json.dumps(sql_execution_output)
        resumed_scratchpad = f"{interrupted_scratchpad}Observation: {sql_result_json_str}\n"
    else:
        error_msg = "Internal error: Unexpected SQL execution result type."
        chat_history.append({"role": "assistant", "content": error_msg})
        request.session['chat_history'] = chat_history
        _clear_pending_sql_state(request)
        return JsonResponse({'error': error_msg})

    try:
        auto_tools = agent_logic.get_auto_execute_tools()
        result = agent_logic.run_agent_invoke_with_tools(
            prompt=original_question,
            chat_history_list_of_dicts=current_chat_history_for_resume,
            agent_scratchpad_str=resumed_scratchpad,
            custom_tools=auto_tools,
        )
        agent_output = result.get('output', str(result))

        chat_history.append({"role": "assistant", "content": agent_output})
        request.session['chat_history'] = chat_history
        _clear_pending_sql_state(request)
        return JsonResponse({'answer': agent_output})

    except agent_logic.SQLApprovalRequired as e_resume:
        if request.session.get('auto_execute_sql'):
            return _execute_sql_and_resume(
                request,
                e_resume.sql,
                original_question,
                e_resume.agent_scratchpad,
                current_chat_history_for_resume,
                chat_history,
            )

        request.session['pending_sql'] = e_resume.sql
        request.session['pending_question'] = original_question
        request.session['pending_tool_input'] = e_resume.tool_input
        request.session['pending_agent_scratchpad'] = e_resume.agent_scratchpad
        request.session['resume_chat_history'] = current_chat_history_for_resume
        return JsonResponse({
            'sql_to_approve': e_resume.sql,
            'tool_input': e_resume.tool_input
        })

    except Exception as e_resume_general:
        error_message = f"Error processing request after SQL approval: {str(e_resume_general)}"
        chat_history.append({"role": "assistant", "content": error_message})
        request.session['chat_history'] = chat_history
        _clear_pending_sql_state(request)
        return JsonResponse({'error': error_message})

def chat_view(request):
    # Clear ALL relevant session state when the main chat page is loaded
    keys_to_clear = [
        'pending_sql', 'pending_question', 'pending_tool_input',
        'pending_agent_scratchpad', 'resume_chat_history', 'chat_history'
    ]
    for key in keys_to_clear:
        if key in request.session:
            del request.session[key]
    # Initialize an empty chat_history for the new session
    request.session['chat_history'] = []
    return render(request, 'chat_app/index.html')

@csrf_exempt # For simplicity in this example. Use proper CSRF for production.
def ask_agent_view(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            user_question = data.get('question')
            approved_sql = data.get('approved_sql')

            if 'auto_execute_sql' in data:
                request.session['auto_execute_sql'] = bool(data['auto_execute_sql'])

            auto_execute_sql = request.session.get('auto_execute_sql', False)

            # Always get chat_history; it's initialized by chat_view or appended to.
            chat_history = request.session.get('chat_history', [])

            if approved_sql: # User is approving/running SQL
                sql_to_execute = approved_sql
                original_question = request.session.pop('pending_question', 'Error: Original question not found in session.')
                interrupted_scratchpad = request.session.pop('pending_agent_scratchpad', '')
                current_chat_history_for_resume = request.session.pop('resume_chat_history', list(chat_history))
                request.session.pop('pending_tool_input', None)

                return _execute_sql_and_resume(
                    request,
                    sql_to_execute,
                    original_question,
                    interrupted_scratchpad,
                    current_chat_history_for_resume,
                    chat_history,
                )

            elif user_question:
                # New question from user
                chat_history.append({"role": "user", "content": user_question})
                request.session['chat_history'] = chat_history 

                # For a new question, scratchpad is empty. History is prior turns in *this session*.
                try:
                    # Pass chat_history[:-1] which is history *before* the current user_question
                    result = agent_logic.run_agent_invoke(
                        prompt=user_question,
                        chat_history_list_of_dicts=chat_history[:-1],
                        agent_scratchpad_str="" # Always empty for a new question
                    )
                    agent_output = result.get('output', str(result))
                    chat_history.append({"role": "assistant", "content": agent_output})
                    request.session['chat_history'] = chat_history

                    _clear_pending_sql_state(request)

                    return JsonResponse({'answer': agent_output})

                except agent_logic.SQLApprovalRequired as e:
                    if auto_execute_sql:
                        request.session['resume_chat_history'] = list(chat_history)
                        return _execute_sql_and_resume(
                            request,
                            e.sql,
                            user_question,
                            e.agent_scratchpad,
                            request.session['resume_chat_history'],
                            chat_history,
                        )

                    request.session['pending_sql'] = e.sql
                    request.session['pending_question'] = user_question
                    request.session['pending_tool_input'] = e.tool_input
                    request.session['pending_agent_scratchpad'] = e.agent_scratchpad
                    # Save chat_history (which includes the current user_question) for resumption
                    request.session['resume_chat_history'] = list(chat_history) # Save a copy for this specific flow

                    return JsonResponse({
                        'sql_to_approve': e.sql,
                        'tool_input': e.tool_input
                    })
            else:
                return JsonResponse({'error': 'No question or approved_sql provided'}, status=400)

        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON in request body'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request method'}, status=405)
