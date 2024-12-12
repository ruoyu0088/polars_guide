document.addEventListener('DOMContentLoaded', () => {
    // ページ内のすべての <span> 要素を取得
    const spans = document.querySelectorAll('span.c1');
  
    // 各 <span> 要素を処理
    spans.forEach(span => {
      const textContent = span.textContent;
  
      // 丸数字 (❶～❾) に一致する場合
      const match = textContent.match(/#(❶|❷|❸|❹|❺|❻|❼|❽|❾)/);
      if (match) {
        // `#`と丸数字を分割し、新しいHTMLを設定
        span.innerHTML = `#<span style="font-size: 200%;">${match[1]}</span>`;
      }
    });
  });