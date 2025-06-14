document.addEventListener('DOMContentLoaded', function() {
  var input = document.getElementById("userInput");

  // Thực hiện hành động khi người dùng nhấn Enter trong trường input
  input.addEventListener("keypress", function(event) {
      // Kiểm tra xem phím được nhấn có phải là Enter không
      if (event.key === "Enter") {
          event.preventDefault(); // Ngăn không cho form submit theo cách mặc định
          predictionMessage(); // Gọi hàm classifyMessage
      }
  });
});

function predictionMessage() {
  var input = document.getElementById("userInput");
  var loadingMessage = document.getElementById('loadingMessage');
  var chatboxView = document.getElementById('chatbox');
  var intro = document.getElementById('intro');
  
  // Ẩn intro
  intro.style.display = 'none';
  // Hiển thị chatbox
  chatboxView.style.display = 'block';
  // Hiển thị thông báo "đang tải"
  loadingMessage.style.display = 'block';
  var message = input.value.trim();
  // Xóa nội dung của input
  input.value = '';
  document.getElementById('charCount').innerText = "0/100";

  if (message) {
    var chatbox = document.getElementById("chatbox");
    const htmlMessage = `
        <div class="chat__conversation-board__message-container">
            <div class="chat__conversation-board__message__person">
                <div class="chat__conversation-board__message__person__avatar"><img width="96" height="96" src="https://img.icons8.com/fluency/96/user-male-circle--v1.png" alt="user-male-circle--v1"/></div><span class="chat__conversation-board__message__person__nickname">Monika Figi</span>
                </div>
                <div class="chat__conversation-board__message__context">
                <div class="chat__conversation-board__message__bubble"> <span>`+ message +`</span></div>
                </div>
                <div class="chat__conversation-board__message__options">
                <button class="btn-icon chat__conversation-board__message__option-button option-item emoji-button">
                    <svg class="feather feather-smile sc-dnqmqq jxshSx" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                    <circle cx="12" cy="12" r="10"></circle>
                    <path d="M8 14s1.5 2 4 2 4-2 4-2"></path>
                    <line x1="9" y1="9" x2="9.01" y2="9"></line>
                    <line x1="15" y1="9" x2="15.01" y2="9"></line>
                    </svg>
                </button>
                <button class="btn-icon chat__conversation-board__message__option-button option-item more-button">
                    <svg class="feather feather-more-horizontal sc-dnqmqq jxshSx" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                    <circle cx="12" cy="12" r="1"></circle>
                    <circle cx="19" cy="12" r="1"></circle>
                    <circle cx="5" cy="12" r="1"></circle>
                    </svg>
                </button>
            </div>
        </div>
        `;
    chatbox.innerHTML += htmlMessage;

    $.post("/prediction_message", { message: message }, function(data) {
        
        
        // Ẩn thông báo "đang tải"
        loadingMessage.style.display = 'none';

        
        
        const htmlPredict = `
        <div class="chat__conversation-board__message-container">
            <div class="chat__conversation-board__message__person">
                <div class="chat__conversation-board__message__person__avatar"><img width="96" height="96" src="https://img.icons8.com/color/96/message-bot.png" alt="message-bot"/></div><span class="chat__conversation-board__message__person__nickname">Monika Figi</span>
                </div>
                <div class="chat__conversation-board__message__context">
                <div class="chat__conversation-board__message__bubble"> 
                    <span>`+ data.prediction +`</span>
                    <span style="display: none;">`+ data.f1scores +`</span>
                </div>
                </div>
                <div class="chat__conversation-board__message__options">
                <button class="btn-icon chat__conversation-board__message__option-button option-item emoji-button">
                    <svg class="feather feather-smile sc-dnqmqq jxshSx" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                    <circle cx="12" cy="12" r="10"></circle>
                    <path d="M8 14s1.5 2 4 2 4-2 4-2"></path>
                    <line x1="9" y1="9" x2="9.01" y2="9"></line>
                    <line x1="15" y1="9" x2="15.01" y2="9"></line>
                    </svg>
                </button>
                <button class="btn-icon chat__conversation-board__message__option-button option-item more-button">
                    <svg class="feather feather-more-horizontal sc-dnqmqq jxshSx" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                    <circle cx="12" cy="12" r="1"></circle>
                    <circle cx="19" cy="12" r="1"></circle>
                    <circle cx="5" cy="12" r="1"></circle>
                    </svg>
                </button>
            </div>
        </div>
        `;
        chatbox.innerHTML += htmlPredict;
        chatbox.scrollTop = chatbox.scrollHeight; // Scroll to the bottom
    });
  }
}

function updateCharacterCount() {
    // Lấy số ký tự từ input
    var inputText = document.getElementById('userInput').value;
    // Đếm số ký tự
    var count = inputText.length;
    // Hiển thị số ký tự
    document.getElementById('charCount').innerText = count + "/100";
}