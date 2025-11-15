from ollama import chat

def ask_user():
    user_answer = str(input("Please enter a query:\n"))
    user_input = {'role': 'user', 'content': user_answer}
    return user_input

def ask_bot(message_list):
    bot_response = chat(
        model = 'gpt-oss:latest',
        messages = message_list,
        think = 'low',
        stream = False
    )
    # print('Thinking:\n', bot_response.message.thinking)
    print('Answer:\n', bot_response.message.content)

def main():
    instruction_message = {'role': 'Support assistant for software dev. Output for Alacritty terminal; format for space efficiency. No follow-ups.', 'content':'Hi!'}
    messages = [instruction_message]
    while True:
        try:
            user_prompt = ask_user()
            messages.append(user_prompt)
            ask_bot(messages)
            
        except KeyboardInterrupt:
            print('Ending.\n')
            exit()

        except Exception as err:
            print(error)

main()
