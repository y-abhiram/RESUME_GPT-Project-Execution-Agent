const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
recognition.lang = 'en-US';
recognition.interimResults = false;

const synth = window.speechSynthesis;

function startListening() {
  recognition.start();
  document.getElementById("status").innerText = "🎤 Listening...";
}

recognition.onresult = async (event) => {
  const query = event.results[0][0].transcript;
  document.getElementById("query").value = query;
  document.getElementById("status").innerText = "🤖 Processing...";
  await sendQuery(query);
};

recognition.onerror = (e) => {
  console.error("Speech recognition error", e);
  document.getElementById("status").innerText = "❌ Speech recognition error.";
};

async function sendQuery(text) {
  try {
    const res = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: text })
    });

    if (!res.ok) {
      console.error("Server error:", res.statusText);
      document.getElementById("responseArea").innerText = "Server error.";
      return;
    }

    const data = await res.json();
    console.log("DEBUG: backend returned →", data);

    if (data && data.chunks && data.chunks.length > 0) {
      displayAndSpeakChunks(data.chunks);
    } else {
      displayAndSpeakChunks(["Sorry, I couldn't find information in your resume data."]);
    }
  } catch (err) {
    console.error("Fetch error:", err);
    document.getElementById("responseArea").innerText = "Network error.";
  }
}

function displayAndSpeakChunks(chunks) {
  const area = document.getElementById("responseArea");
  area.innerHTML = "";
  const synth = window.speechSynthesis;

  if (!chunks || chunks.length === 0) {
    area.innerHTML = "<p>Agent: No response.</p>";
    return;
  }

  let i = 0;
  function speakNext() {
    if (i >= chunks.length) {
      document.getElementById("status").innerText = "✅ Done.";
      return;
    }

    const text = chunks[i];
    if (!text || text === "undefined" || text.trim() === "") {
      i++;
      speakNext();
      return;
    }

    const p = document.createElement("p");
    p.className = "assistant-chunk";
    p.innerText = "Agent: " + text;
    area.appendChild(p);
    p.scrollIntoView({ behavior: "smooth", block: "end" });

    const utter = new SpeechSynthesisUtterance(text);
    utter.rate = 1;
    utter.pitch = 1;
    utter.onend = () => {
      i++;
      setTimeout(speakNext, 400);
    };
    synth.speak(utter);
  }

  document.getElementById("status").innerText = "🗣️ Speaking...";
  speakNext();
}

async function submitText() {
  const text = document.getElementById("query").value.trim();
  if (!text) return;
  document.getElementById("status").innerText = "🤖 Processing...";
  await sendQuery(text);
}

