const GEM_ORDER = ["black", "white", "red", "blue", "green"];
const GEM_ORDER_WITH_GOLD = ["black", "white", "red", "blue", "green", "gold"];
const GEM_LABEL = { black: "黑曜", white: "钻石", red: "红宝", blue: "蓝宝", green: "祖母绿", gold: "黄金" };
const GEM_COLORS_CN = { black: "黑", white: "白", red: "红", blue: "蓝", green: "绿", gold: "金" };

let state = null;
let selection = []; // list of color strings chosen for a take action
let returnSelection = []; // list of color strings chosen to return (gem return flow)

const boardEl = document.getElementById("board");
const statusBarEl = document.getElementById("statusBar");
const overlayEl = document.getElementById("overlay");
const overlayTitleEl = document.getElementById("overlayTitle");
const overlayButtonsEl = document.getElementById("overlayButtons");
const gameOverEl = document.getElementById("gameOverBanner");
const gameOverTextEl = document.getElementById("gameOverText");
const returnOverlayEl = document.getElementById("returnOverlay");
const returnInfoEl = document.getElementById("returnInfo");
const returnGemSelectionEl = document.getElementById("returnGemSelection");
const returnStatusEl = document.getElementById("returnStatus");
const returnConfirmBtn = document.getElementById("returnConfirmBtn");
const returnCancelBtn = document.getElementById("returnCancelBtn");
const gameLogEntriesEl = document.getElementById("gameLogEntries");

document.getElementById("newGameBtn").addEventListener("click", startNewGame);
document.getElementById("playAgainBtn").addEventListener("click", startNewGame);
document.getElementById("passBtn").addEventListener("click", () => sendAction(50));
document.getElementById("overlayCancel").addEventListener("click", closeOverlay);
returnCancelBtn.addEventListener("click", cancelReturn);
returnConfirmBtn.addEventListener("click", confirmReturn);

async function startNewGame() {
  const humanFirst = document.getElementById("firstPlayerSelect").value === "true";
  gameOverEl.classList.add("hidden");
  returnOverlayEl.classList.add("hidden");
  selection = [];
  returnSelection = [];
  const res = await fetch("/api/new_game", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ human_first: humanFirst }),
  });
  state = await res.json();
  checkNeedsReturn();
  render();
}

async function sendAction(action) {
  if (!state || !state.is_human_turn) return;
  if (state.needs_return) return; // must resolve return first
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
  checkNeedsReturn();
  render();
}

async function confirmReturn() {
  if (!state || !state.needs_return) return;
  const body = {
    action: state.pending_action,
    return_colors: returnSelection,
  };
  const res = await fetch("/api/return_gems", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json();
    alert(err.error || "退还失败");
    return;
  }
  state = await res.json();
  returnSelection = [];
  returnOverlayEl.classList.add("hidden");
  render();
}

async function cancelReturn() {
  returnSelection = [];
  returnOverlayEl.classList.add("hidden");
  // Cancel via dedicated endpoint — no action consumed
  const res = await fetch("/api/cancel_return", { method: "POST" });
  if (res.ok) {
    state = await res.json();
    render();
  }
}

function checkNeedsReturn() {
  if (state && state.needs_return) {
    returnSelection = [];  // only reset when first entering return flow
    renderReturnOverlay();
  }
}

function renderReturnOverlay() {
  const excess = state.excess_count;
  const currentTokens = state.current_tokens;
  const tokensAfter = state.tokens_after_take;
  const gemsTaken = state.gems_taken || {};

  // Info text
  let takenText = [];
  for (const [color, count] of Object.entries(gemsTaken)) {
    if (count > 0) takenText.push(`${GEM_LABEL[color]}+${count}`);
  }
  returnInfoEl.innerHTML = `拿取宝石: ${takenText.join(", ")}<br>当前手牌 ${Object.values(currentTokens).reduce((a,b)=>a+b,0)} → 拿取后 ${Object.values(tokensAfter).reduce((a,b)=>a+b,0)}<br>需要退还 <b>${excess}</b> 颗宝石（点击宝石选择退还）`;

  // Gem selection
  returnGemSelectionEl.innerHTML = "";
  GEM_ORDER_WITH_GOLD.forEach(color => {
    const available = tokensAfter[color] || 0;
    if (available <= 0) return;

    const div = document.createElement("div");
    const selCount = returnSelection.filter(c => c === color).length;
    div.className = `return-gem gem-${color}` + (selCount > 0 ? " selected" : "");
    div.innerHTML = `
      <div class="gem-count">${available}</div>
      <div class="gem-label">${GEM_LABEL[color]}</div>
      ${selCount > 0 ? `<div class="return-badge">${selCount}</div>` : ""}
    `;
    div.addEventListener("click", () => {
      if (returnSelection.length >= excess) {
        // If already at max, remove a selection of this color first if any
        const idx = returnSelection.indexOf(color);
        if (idx >= 0) {
          returnSelection.splice(idx, 1);
        } else {
          return; // can't add more
        }
      } else {
        returnSelection.push(color);
      }
      renderReturnOverlay();
    });
    returnGemSelectionEl.appendChild(div);
  });

  // Status
  if (returnSelection.length === excess) {
    returnStatusEl.textContent = `将退还: ${returnSelection.map(c => GEM_LABEL[c]).join(", ")}`;
    returnConfirmBtn.disabled = false;
  } else {
    returnStatusEl.textContent = `还需选择 ${excess - returnSelection.length} 颗`;
    returnConfirmBtn.disabled = true;
  }

  returnOverlayEl.classList.remove("hidden");
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
  if (state.needs_return) return;

  const len = selection.length;
  const distinct = new Set(selection);

  // Count how many non-gold colors have gems available on the board
  const availableColors = GEM_ORDER.filter(c => (state.gems_available[c] || 0) > 0);
  const canTake3 = availableColors.length >= 3;

  if (len === 0) {
    // First gem — could be take1, or start of take2/take3
    selection = [color];
  } else if (len === 1) {
    if (selection[0] === color) {
      // Same color → going for take2 (needs >=4 in pool)
      selection = [color, color];
    } else if (canTake3) {
      // Different color + enough colors on board → enter take3 path
      selection = [selection[0], color];
    } else {
      // Not enough colors for take3 → switch to new color (take1)
      selection = [color];
    }
  } else if (len === 2) {
    if (distinct.size === 1) {
      // Already have 2 of same color — at max for take2
      return;
    } else {
      // Have 2 different colors — on take3 path, only allow 3rd different
      if (distinct.has(color)) return; // can't repeat on take3 path
      if (selection.length >= 3) return;
      selection = [...selection, color];
    }
  } else if (len === 3) {
    return; // max 3 gems
  }

  refreshGemDisplay();
  renderGemBar();
}

// Update existing gem tokens' visual state without destroying/recreating them
function refreshGemDisplay() {
  const row = boardEl.querySelector(".gems-row");
  if (!row) return;
  const gems = state.gems_available;
  const tokens = row.querySelectorAll(".gem-token");
  tokens.forEach(tok => {
    const color = Array.from(tok.classList)
      .find(c => c.startsWith("gem-") && c !== "gem-token")
      ?.replace("gem-", "");
    if (!color) return;
    const count = gems[color];
    const selCount = selection.filter(c => c === color).length;
    const canClick = color !== "gold" && state.is_human_turn && !state.needs_return && count > 0;
    // Rebuild className
    const classes = ["gem-token", `gem-${color}`];
    if (selCount > 0) classes.push("selected");
    if (!canClick) classes.push("disabled");
    tok.className = classes.join(" ");
    // Update count and badge
    tok.innerHTML = `<div class="count">${count}</div>` + (selCount > 0 ? `<div class="sel-badge">${selCount}</div>` : "");
    // Re-attach click handler (cleared by innerHTML assignment)
    if (canClick) {
      tok.addEventListener("click", () => onGemClick(color));
    }
  });
}

function renderGems() {
  const existing = boardEl.querySelector(".gems-row");

  const row = document.createElement("div");
  row.className = "gems-row";
  const gems = state.gems_available;
  GEM_ORDER_WITH_GOLD.forEach(color => {
    const count = gems[color];
    const div = document.createElement("div");
    const selCount = selection.filter(c => c === color).length;
    const canClick = color !== "gold" && state.is_human_turn && !state.needs_return && count > 0;
    const classes = [`gem-token`, `gem-${color}`];
    if (selCount > 0) classes.push("selected");
    if (!canClick) classes.push("disabled");
    div.className = classes.join(" ");
    div.innerHTML = `<div class="count">${count}</div>` + (selCount > 0 ? `<div class="sel-badge">${selCount}</div>` : "");
    if (canClick) {
      div.addEventListener("click", () => onGemClick(color));
    }
    row.appendChild(div);
  });

  if (existing) {
    existing.replaceWith(row);
  } else {
    boardEl.appendChild(row);
  }
}

function renderGemBar() {
  let bar = document.getElementById("gemBar");
  if (bar) bar.remove();

  if (selection.length === 0) return;

  bar = document.createElement("div");
  bar.id = "gemBar";
  bar.className = "gem-bar";

  const label = document.createElement("span");
  label.className = "gem-bar-label";
  const names = selection.map(c => GEM_LABEL[c]).join(" + ");
  label.textContent = `已选: ${names}`;
  bar.appendChild(label);

  const match = selectionMatch();
  if (match) {
    const confirmBtn = document.createElement("button");
    confirmBtn.textContent = "确认拿取";
    confirmBtn.addEventListener("click", () => sendAction(match.action));
    bar.appendChild(confirmBtn);
  }

  const clearBtn = document.createElement("button");
  clearBtn.textContent = "清空";
  clearBtn.addEventListener("click", () => {
    selection = [];
    renderGems();       // full rebuild (uses replaceWith, keeps position)
    renderGemBar();     // removes bar since selection is empty
  });
  bar.appendChild(clearBtn);

  const gemsRow = boardEl.querySelector(".gems-row");
  if (gemsRow) {
    gemsRow.insertAdjacentElement("afterend", bar);
  }
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
    const legal = card && state.is_human_turn && !state.needs_return && (buyAction || reserveAction);
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

  // ---- Gem Grid (6 columns) with bonus squares ----
  const totalTokens = GEM_ORDER_WITH_GOLD.reduce((sum, c) => sum + (p.tokens[c] || 0), 0);

  // Count bonus squares per color from purchased cards
  const bonusSquares = { black: 0, white: 0, red: 0, blue: 0, green: 0 };
  p.purchased.forEach(card => {
    if (card.bonus && card.bonus !== "gold") {
      bonusSquares[card.bonus] = (bonusSquares[card.bonus] || 0) + 1;
    }
  });

  html += `<div class="gem-grid">`;

  GEM_ORDER_WITH_GOLD.forEach(color => {
    const tokenCount = p.tokens[color] || 0;
    html += `<div class="gem-col">`;
    html += `<div class="col-label">${GEM_COLORS_CN[color]}</div>`;
    html += `<div class="col-tokens">${tokenCount}</div>`;

    // Bonus squares row (max 5 fit in one row)
    if (color === "gold") {
      html += `<div class="col-bonuses" style="opacity:0.3">—</div>`;
    } else {
      const n = bonusSquares[color] || 0;
      html += `<div class="col-bonus-squares">`;
      for (let i = 0; i < n; i++) {
        html += `<span class="bonus-square bg-${color}"></span>`;
      }
      // Empty placeholders to maintain height if 0 bonuses
      if (n === 0) {
        html += `<span class="bonus-square bonus-empty"></span>`;
      }
      html += `</div>`;
    }
    html += `</div>`;
  });

  html += `</div>`;

  // Total count
  html += `<div class="gem-grid-total">手牌总计: <span class="total-value">${totalTokens}</span></div>`;

  panel.innerHTML = html;

  // Reserved cards
  const reservedWrap = document.createElement("div");
  reservedWrap.className = "reserved-cards";
  if (key === "human") {
    (p.reserved || []).forEach((card, slot) => {
      const buyAction = findAction("buy_reserved", a => a.slot === slot);
      const el = renderCard(card, buyAction && !state.needs_return ? "legal" : "", (buyAction && !state.needs_return) ? () =>
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

// ---- Game Log ----

function renderGameLog() {
  const log = state.ai_action_log || [];
  gameLogEntriesEl.innerHTML = "";

  if (log.length === 0) {
    gameLogEntriesEl.innerHTML = `<div class="game-log-empty">暂无记录<br>开始新的一局吧</div>`;
    return;
  }

  log.forEach(entry => {
    const div = document.createElement("div");
    div.className = `game-log-entry ${entry.player === "ai" ? "entry-ai" : "entry-human"}`;

    let icon = "";
    if (entry.category === "pass") icon = "⏭️";
    else if (entry.category && entry.category.startsWith("buy")) icon = "💰";
    else if (entry.category && entry.category.startsWith("reserve")) icon = "📋";
    else if (entry.category && entry.category.startsWith("take")) icon = "💎";
    else icon = "•";

    const playerLabel = entry.player === "ai" ? "AI" : "你";

    let html = `<span class="entry-turn">T${entry.turn_number}</span>`;
    html += `<span class="entry-icon">${icon}</span>`;
    html += `<span class="entry-player">[${playerLabel}]</span> `;
    html += `<span class="entry-desc">${entry.description}</span>`;

    // Card info for buy/reserve
    if (entry.card_info) {
      const ci = entry.card_info;
      html += `<div class="entry-card">`;
      html += `<span class="card-bonus-dot bg-${ci.bonus}"></span>`;
      html += `<span>Lv${ci.level} +${ci.points}pt</span>`;
      // Show cost
      const costParts = [];
      for (const [c, n] of Object.entries(ci.cost || {})) {
        if (n > 0) costParts.push(`${GEM_COLORS_CN[c]}${n}`);
      }
      if (costParts.length > 0) {
        html += `<span style="font-size:10px;opacity:0.7">(${costParts.join(" ")})</span>`;
      }
      html += `</div>`;
    }

    // Return info
    if (entry.returned_gems && entry.returned_gems.length > 0) {
      html += `<div style="font-size:10px;color:#ff6b6b;margin-top:2px">退还: ${entry.returned_gems.map(c => GEM_LABEL[c]).join(", ")}</div>`;
    }

    div.innerHTML = html;
    gameLogEntriesEl.appendChild(div);
  });

  // Auto-scroll to bottom
  gameLogEntriesEl.scrollTop = gameLogEntriesEl.scrollHeight;
}

// ---- Status Bar ----

function renderStatusBar() {
  let text = "";
  if (!state.game_over) {
    if (state.needs_return) {
      text = "⚠️ 请选择要退还的宝石";
    } else {
      text = state.is_human_turn ? "轮到你了" : "AI 思考中/已行动";
    }
    if (state.final_round_flag) text += "　【最后一轮】";
  }
  statusBarEl.innerHTML = text;
}

// ---- Main Render ----

function render() {
  boardEl.innerHTML = "";
  if (!state) return;

  renderStatusBar();
  renderGems();
  renderGemBar();

  [3, 2, 1].forEach(level => renderLevelRow(level));

  const playersRow = document.createElement("div");
  playersRow.className = "players-row";
  playersRow.appendChild(renderPlayerPanel("human", "你"));
  playersRow.appendChild(renderPlayerPanel("ai", "AI"));
  boardEl.appendChild(playersRow);

  document.getElementById("passBtn").disabled = !state.is_human_turn || state.needs_return;

  renderGameLog();

  if (state.game_over) {
    const map = { human: "你赢了！", ai: "AI 赢了。", draw: "平局。" };
    gameOverTextEl.textContent = map[state.winner] || "对局结束";
    gameOverEl.classList.remove("hidden");
  }
}
