const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
recognition.lang = 'en-US';
recognition.interimResults = false;

function startListening() {
  recognition.start();
}

recognition.onresult = async (event) => {
  const query = event.results[0][0].transcript;
  document.getElementById("query").value = query;
  sendQuery(query);
};

function speak(text) {
  const synth = window.speechSynthesis;
  const utter = new SpeechSynthesisUtterance(text);
  synth.speak(utter);
}

