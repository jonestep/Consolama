from ollama import chat
import tempfile
import subprocess
import os

# Configuration
MODEL = 'gpt-oss:latest'
MAX_CONTEXT_TOKENS = 8000  # Adjust based on your model's context window
RECENT_TURNS_TO_KEEP = 6  # Number of recent message pairs (user+assistant) to keep uncompressed
COMPRESSION_THRESHOLD = 0.85  # Compress when context reaches 85% of max

def estimate_tokens(text):
    """Rough token estimation: ~4 characters per token for English text"""
    if not text:
        return 0
    return len(str(text)) // 4

def count_context_tokens(messages):
    """Estimate total tokens in message list"""
    total = 0
    for msg in messages:
        # Count tokens for role and content
        total += estimate_tokens(msg.get('role', ''))
        total += estimate_tokens(msg.get('content', ''))
        # Add overhead for message structure (~10 tokens per message)
        total += 10
    return total

def compress_messages(messages, recent_turns_to_keep=RECENT_TURNS_TO_KEEP):
    """
    Compress older messages using semantic summarization.
    Keeps the system message, recent turns, and compresses everything in between.
    """
    if len(messages) <= recent_turns_to_keep * 2 + 1:  # +1 for system message
        return messages
    
    system_msg = messages[0] if messages[0].get('role') == 'system' else None
    recent_messages = messages[-(recent_turns_to_keep * 2):] if recent_turns_to_keep > 0 else []
    
    # Messages to compress (everything between system and recent)
    if system_msg:
        messages_to_compress = messages[1:-(recent_turns_to_keep * 2)] if recent_turns_to_keep > 0 else messages[1:]
    else:
        messages_to_compress = messages[:-(recent_turns_to_keep * 2)] if recent_turns_to_keep > 0 else messages
    
    if not messages_to_compress:
        return messages
    
    # Create a summary prompt
    conversation_text = "\n".join([
        f"{msg.get('role', 'unknown').upper()}: {msg.get('content', '')}"
        for msg in messages_to_compress
    ])
    
    summary_prompt = {
        'role': 'user',
        'content': f"""Please provide a concise summary of the following conversation history. 
Focus on key topics, decisions, and important context that should be remembered for future reference.
Keep it brief but comprehensive:

{conversation_text}

Summary:"""
    }
    
    try:
        # Use the model to summarize
        summary_response = chat(
            model=MODEL,
            messages=[summary_prompt],
            think='low',
            stream=False
        )
        
        summary_content = summary_response.message.content
        summary_msg = {'role': 'assistant', 'content': f"[Previous conversation summary]: {summary_content}"}
        
        # Reconstruct message list
        compressed_messages = []
        if system_msg:
            compressed_messages.append(system_msg)
        compressed_messages.append(summary_msg)
        compressed_messages.extend(recent_messages)
        
        return compressed_messages
    except Exception as e:
        print(f"Warning: Compression failed ({e}), keeping original messages")
        return messages

def manage_context(messages):
    """
    Check context size and compress if needed.
    Returns the (possibly compressed) message list.
    """
    current_tokens = count_context_tokens(messages)
    
    if current_tokens >= MAX_CONTEXT_TOKENS * COMPRESSION_THRESHOLD:
        print(f"[Context: {current_tokens}/{MAX_CONTEXT_TOKENS} tokens] Compressing older messages...")
        compressed = compress_messages(messages)
        new_tokens = count_context_tokens(compressed)
        print(f"[Context after compression: {new_tokens}/{MAX_CONTEXT_TOKENS} tokens]")
        return compressed
    
    return messages

def ask_user():
    user_answer = str(input("Please enter a query:\n"))
    user_input = {'role': 'user', 'content': user_answer}
    return user_input

def format_messages_for_display(messages):
    """Format messages list into a readable string representation"""
    formatted = []
    for i, msg in enumerate(messages):
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')
        formatted.append(f"[{i}] {role.upper()}: {content}")
    return '\n\n'.join(formatted)

def parse_messages_from_text(text):
    """Parse formatted text back into messages list"""
    lines = text.strip().split('\n\n')
    messages = []
    
    for block in lines:
        if not block.strip():
            continue
        
        # Try to parse as [index] ROLE: content
        if ':' in block:
            parts = block.split(':', 1)
            if len(parts) == 2:
                header = parts[0].strip()
                content = parts[1].strip()
                
                # Extract role from header (e.g., "[0] SYSTEM" -> "system")
                if ']' in header:
                    role_part = header.split(']', 1)[1].strip().lower()
                    role = role_part
                else:
                    role_part = header.strip().lower()
                    role = role_part
                
                messages.append({'role': role, 'content': content})
    
    return messages

def get_default_editor():
    """Get the system's default text editor"""
    # Check common environment variables
    editor = os.environ.get('EDITOR')
    if editor:
        return editor.split()[0]  # Take first word in case of "editor --args"
    
    # Fallback to common editors
    common_editors = ['nano', 'vim', 'vi', 'gedit', 'kate', 'code', 'subl']
    for ed in common_editors:
        if subprocess.run(['which', ed], capture_output=True).returncode == 0:
            return ed
    
    # Last resort
    return 'nano'

def show_conversation_editor(messages):
    """
    Show conversation history in system's default text editor.
    Returns the updated messages list, or original if cancelled.
    """
    # Format messages for editing
    formatted_text = format_messages_for_display(messages)
    
    # Add header instructions
    header = """# Conversation History Editor
# Edit the conversation below. Format: [index] ROLE: content
# Save and close the editor to update, or exit without saving to cancel.
# Lines starting with # are comments and will be ignored.

"""
    full_text = header + formatted_text
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(full_text)
        temp_file = f.name
    
    try:
        # Get editor
        editor = get_default_editor()
        
        print(f"\n[Opening conversation history in {editor}...]")
        print("[Edit the file, save and close to update, or exit without saving to cancel]")
        
        # Open editor and wait for it to close
        result = subprocess.run([editor, temp_file])
        
        # Read the edited file
        with open(temp_file, 'r') as f:
            edited_text = f.read()
        
        # Remove header comments and empty lines
        lines = edited_text.split('\n')
        content_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith('#'):
                content_lines.append(line)
        
        edited_content = '\n'.join(content_lines)
        
        if not edited_content.strip():
            print("[No content found, keeping original conversation history]")
            return messages
        
        # Parse the edited content
        try:
            parsed_messages = parse_messages_from_text(edited_content)
            
            if not parsed_messages:
                print("[Warning: Could not parse conversation history, keeping original]")
                return messages
            
            print(f"[Conversation history updated: {len(parsed_messages)} messages]")
            return parsed_messages
            
        except Exception as e:
            print(f"[Error parsing conversation history: {e}, keeping original]")
            return messages
            
    except KeyboardInterrupt:
        print("\n[Editor cancelled, keeping original conversation history]")
        return messages
    except Exception as e:
        print(f"[Error opening editor: {e}, keeping original conversation history]")
        return messages
    finally:
        # Clean up temp file
        try:
            os.unlink(temp_file)
        except:
            pass

def ask_bot(message_list):
    bot_response = chat(
        model=MODEL,
        messages=message_list,
        think='low',
        stream=False
    )
    # print('Thinking:\n', bot_response.message.thinking)
    print('Answer:\n', bot_response.message.content)
    # Store the assistant's response in the conversation history
    assistant_message = {'role': 'assistant', 'content': bot_response.message.content}
    return assistant_message

def main():
    instruction_message = {
        'role': 'system', 
        'content': 'You are a support assistant for software development. Output for Alacritty terminal; format for space efficiency.'
    }
    messages = [instruction_message]
    
    while True:
        try:
            user_prompt = ask_user()
            messages.append(user_prompt)
            
            # Manage context before asking bot
            messages = manage_context(messages)
            
            assistant_response = ask_bot(messages)
            messages.append(assistant_response)  # Add bot's response to maintain conversation history
            
            # Show conversation history editor popup
            print("\n[Opening conversation history editor...]")
            updated_messages = show_conversation_editor(messages)
            if updated_messages:
                messages = updated_messages
                print("[Conversation history updated]\n")
            
        except KeyboardInterrupt:
            print('Ending.\n')
            exit()

        except Exception as err:
            print(err)

main()
