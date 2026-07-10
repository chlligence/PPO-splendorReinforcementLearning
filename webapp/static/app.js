const GEM_ORDER = ["black", "white", "red", "blue", "green"];
const GEM_LABEL = { black: "黑曜", white: "钻石", red: "红宝", blue: "蓝宝", green: "祖母绿", gold: "黄金" };

let state = null;
let selection = []; // list of color strings chosen for a take action

const boardEl = document.getElementById("board");
const statusBarEl = document.getElementById("statusBar");
const overlayEl = document.getElementById("overlay");
const overlayTitleEl = document.getElementById("overlayTitle");
const overlayButtonsEl = document.getElementById("overlayButtons");
const gameOverEl = document.getElementById("gameOverBanner");
const gameOverTextEl = document.getElementById("gameOverText");

document.getElementById("newGameBtn").addEventListener("click", startNewGame);
document.getElementById("playAgainBtn").addEventListener("click", startNewGame);
document.getElementById("passBtn").addEventListener("click", () => sendAction(50));
document.getElementById("overlayCancel").addEventListener("click", closeOverlay);

async function startNewGame() {
  const humanFirst = document.getElementById("firstPlayerSelect").value === "true";
  gameOverEl.classList.add("hidden");
  selection = [];
  const res = await fetch("/api/new_game", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ human_first: humanFirst }),
  });
  state = await res.json();
  render();
}

async function sendAction(action) {
  if (!state || !state.is_human_turn) return;
  selection = [];
  const res = await fetch("/api/action", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action }),
  });
  if (!res.ok) {
    const err = await res.json();
    console.error(err);
    return;
  }
  state = await res.json();
  closeOverlay();
  render();
}

function findAction(category, pred) {
  if (!state) return null;
  return (state.legal_actions || []).find(a => a.category === category && pred(a)) || null;
}

function closeOverlay() {
  overlayEl.classList.add("hidden");
}

function showOverlay(title, buttons) {
  overlayTitleEl.textContent = title;
  overlayButtonsEl.innerHTML = "";
  buttons.forEach(b => {
    const btn = document.createElement("button");
    btn.textContent = b.label;
    btn.addEventListener("click", () => sendAction(b.action));
    overlayButtonsEl.appendChild(btn);
  });
  overlayEl.classList.remove("hidden");
}

function costToPips(cost) {
  return GEM_ORDER.filter(c => cost[c] > 0)
    .map(c => `<span class="cost-pip bg-${c}">${cost[c]}</span>`)
    .join("");
}

function renderCard(card, extraClass, onClick) {
  const div = document.createElement("div");
  if (!card) {
    div.className = "card empty";
    return div;
  }
  div.className = `card ${extraClass || ""}`;
  div.innerHTML = `
    <div class="top-bar bg-${card.bonus}"><span>Lv${card.level}</span><span>${card.points > 0 ? card.points : ""}</span></div>
    <div class="cost">${costToPips(card.cost)}</div>
  `;
  if (onClick) div.addEventListener("click", onClick);
  return div;
}

function selectionMatch() {
  if (selection.length === 0) return null;
  const counts = {};
  selection.forEach(c => (counts[c] = (counts[c] || 0) + 1));
  const distinct = Object.keys(counts);
  if (selection.length === 1) {
    return findAction("take1", a => a.colors[0] === distinct[0]);
  }
  if (selection.length === 2) {
    if (distinct.length === 1) {
      return findAction("take2", a => a.colors[0] === distinct[0]);
    }
    return null;
  }
  if (selection.length === 3 && distinct.length === 3) {
    const set = new Set(distinct);
    return findAction("take3", a => a.colors.length === 3 && a.colors.every(c => set.has(c)));
  }
  return null;
}

function onGemClick(color) {
  if (!state || !state.is_human_turn) return;
  const attempt = [...selection, color];
  const countInAttempt = attempt.filter(c => c === color).length;
  if (countInAttempt > 2) return;
  if (attempt.length > 3) return;
  selection = attempt;
  renderTakePanel();
  renderGems();
}

function renderGems() {
  const row = document.createElement("div");
  row.className = "gems-row";
  const gems = state.gems_available;
  [...GEM_ORDER, "gold"].forEach(color => {
    const count = gems[color];
    const div = document.createElement("div");
    const selCount = selection.filter(c => c === color).length;
    div.className = `gem-token gem-${color}` + (selCount > 0 ? " selected" : "") + (count === 0 || color === "gold" ? " disabled" : "");
    div.innerHTML = `<div class="count">${count}</div>` + (selCount > 0 ? `<div class="sel-badge">${selCount}</div>` : "");
    if (color !== "gold" && state.is_human_turn && count > 0) {
      div.addEventListener("click", () => onGemClick(color));
    }
    row.appendChild(div);
  });
  boardEl.appendChild(row);
}

function renderTakePanel() {
  let panel = document.getElementById("takePanel");
  if (panel) panel.remove();
  panel = document.createElement("div");
  panel.id = "takePanel";
  panel.className = "take-panel";

  const match = selectionMatch();
  const clearBtn = document.createElement("button");
  clearBtn.textContent = "清空选择";
  clearBtn.addEventListener("click", () => {
    selection = [];
    render();
  });
  panel.appendChild(clearBtn);

  if (selection.length > 0) {
    const label = document.createElement("span");
    label.textContent = match ? `确认: ${match.description}` : "组合不合法（3种不同各1个 / 同色2个 / 单色1个）";
    panel.appendChild(label);
  }

  if (match) {
    const confirmBtn = document.createElement("button");
    confirmBtn.textContent = "确认拿取";
    confirmBtn.addEventListener("click", () => sendAction(match.action));
    panel.appendChild(confirmBtn);
  }

  const gemsRow = boardEl.querySelector(".gems-row");
  gemsRow.insertAdjacentElement("afterend", panel);
}

function renderLevelRow(level) {
  const row = document.createElement("div");
  row.className = "level-row";

  const deckAction = findAction("reserve_deck", a => a.level === level);
  const deck = document.createElement("div");
  deck.className = "deck-back" + (deckAction ? "" : " disabled");
  deck.innerHTML = `<div class="label">L${level}<br/>剩${state.deck_counts[level]}</div>`;
  if (deckAction) {
    deck.addEventListener("click", () =>
      showOverlay(`从等级 ${level} 牌堆盲拿一张预留`, [{ label: "确认预留", action: deckAction.action }])
    );
  }
  row.appendChild(deck);

  const cardsWrap = document.createElement("div");
  cardsWrap.className = "cards-row";
  state.face_up[level].forEach((card, slot) => {
    const buyAction = findAction("buy_face", a => a.level === level && a.slot === slot);
    const reserveAction = findAction("reserve_face", a => a.level === level && a.slot === slot);
    const legal = card && state.is_human_turn && (buyAction || reserveAction);
    const el = renderCard(card, legal ? "legal" : "", legal ? () => {
      const buttons = [];
      if (buyAction) buttons.push({ label: "购买", action: buyAction.action });
      if (reserveAction) buttons.push({ label: "预留", action: reserveAction.action });
      showOverlay(`等级 ${level} 卡牌`, buttons);
    } : null);
    cardsWrap.appendChild(el);
  });
  row.appendChild(cardsWrap);

  boardEl.appendChild(row);
}

function renderPlayerPanel(key, title) {
  const p = state.players[key];
  const panel = document.createElement("div");
  const isActive = (key === "human" && state.is_human_turn) || (key === "ai" && !state.is_human_turn && !state.game_over);
  panel.className = "player-panel" + (isActive ? " active-turn" : "");

  let html = `<h3><span>${title}</span><span>${p.points} 分 / ${p.card_count} 张</span></h3>`;

  html += `<div class="mini-row">`;
  [...GEM_ORDER, "gold"].forEach(c => {
    if (p.tokens[c] > 0) html += `<span class="mini-chip bg-${c}">${GEM_LABEL[c]} ${p.tokens[c]}</span>`;
  });
  html += `</div>`;

  html += `<div class="mini-row">`;
  GEM_ORDER.forEach(c => {
    if (p.bonuses[c] > 0) html += `<span class="mini-chip bg-${c}">${GEM_LABEL[c]}加成 ${p.bonuses[c]}</span>`;
  });
  html += `</div>`;

  panel.innerHTML = html;

  const purchasedWrap = document.createElement("div");
  purchasedWrap.className = "purchased-cards";
  p.purchased.forEach(card => {
    const mini = document.createElement("div");
    mini.style.width = "26px";
    mini.style.height = "26px";
    mini.style.borderRadius = "4px";
    mini.className = `bg-${card.bonus}`;
    mini.title = `Lv${card.level} +${card.points}pt`;
    purchasedWrap.appendChild(mini);
  });
  panel.appendChild(purchasedWrap);

  const reservedWrap = document.createElement("div");
  reservedWrap.className = "reserved-cards";
  if (key === "human") {
    (p.reserved || []).forEach((card, slot) => {
      const buyAction = findAction("buy_reserved", a => a.slot === slot);
      const el = renderCard(card, buyAction ? "legal" : "", buyAction ? () =>
        showOverlay("购买这张预留卡？", [{ label: "购买", action: buyAction.action }]) : null);
      el.style.width = "60px";
      el.style.height = "80px";
      reservedWrap.appendChild(el);
    });
  } else {
    for (let i = 0; i < p.reserved_count; i++) {
      const back = document.createElement("div");
      back.className = "reserved-back";
      back.textContent = "?";
      reservedWrap.appendChild(back);
    }
  }
  panel.appendChild(reservedWrap);

  return panel;
}

function renderStatusBar() {
  let text = "";
  if (!state.game_over) {
    text = state.is_human_turn ? "轮到你了" : "AI 思考中/已行动";
    if (state.final_round_flag) text += "　【最后一轮】";
  }
  if (state.last_ai_actions && state.last_ai_actions.length > 0) {
    text += `　<span class="ai-log">AI 本回合动作: ${state.last_ai_actions.join(" → ")}</span>`;
  }
  statusBarEl.innerHTML = text;
}

function render() {
  boardEl.innerHTML = "";
  if (!state) return;

  renderStatusBar();
  renderGems();
  renderTakePanel();

  [3, 2, 1].forEach(level => renderLevelRow(level));

  const playersRow = document.createElement("div");
  playersRow.className = "players-row";
  playersRow.appendChild(renderPlayerPanel("human", "你"));
  playersRow.appendChild(renderPlayerPanel("ai", "AI"));
  boardEl.appendChild(playersRow);

  document.getElementById("passBtn").disabled = !state.is_human_turn;

  if (state.game_over) {
    const map = { human: "你赢了！", ai: "AI 赢了。", draw: "平局。" };
    gameOverTextEl.textContent = map[state.winner] || "对局结束";
    gameOverEl.classList.remove("hidden");
  }
}
