#ifndef UI_FORM_HPP
#define UI_FORM_HPP

#include <opencv2/opencv.hpp>
#include <graphics.h>

namespace chr {
	class ui_base {
	protected:
		POINT pos_;
		int width_, height_;
		COLORREF fill_, line_;
	public:
		ui_base(POINT position, int width, int height, COLORREF fill = 0xF0F0F0, COLORREF line = 0);
		POINT pos() const { return pos_; }
		int width() const { return width_; }
		int height() const { return height_; }
		virtual void input(const ExMessage& msg) = 0;
		virtual void draw() = 0;
	protected:
		void basic_draw() const;
	};
	class button :public ui_base {
	private:
		std::function<void()> func_;
		std::wstring text_;
		COLORREF text_color_;
		bool is_hovered_;
		bool is_pressed_;
		COLORREF normal_color_;
		COLORREF hover_color_;
		COLORREF press_color_;
	public:
		button(POINT position, int width, int height, COLORREF fill = 0xF0F0F0, COLORREF line = 0);
		void set_text(const std::wstring& text) { text_ = text; }
		void set_function(std::function<void()> func) { func_ = func; }
		void set_text_color(COLORREF color) { text_color_ = color; }
		void set_colors(COLORREF normal, COLORREF hover, COLORREF press);
		void draw() override;
		void input(const ExMessage& msg) override;
		bool is_hovered() const { return is_hovered_; }
		bool is_pressed() const { return is_pressed_; }
	};
	class canvas :public ui_base {
	private:
		bool is_rubber_;
		bool is_lbutton_;
		COLORREF brush_;
		std::list<IMAGE> buffer_;
		std::list<IMAGE> undo_buffer_;
	public:
		canvas(POINT position, int width, int height, COLORREF fill = 0xF0F0F0, COLORREF line = 0);
		const IMAGE& image() { return buffer_.front(); }
		void set_image(const IMAGE& image);
		void set_brush(COLORREF brush) { brush_ = brush; }
		void draw() override;
		void undo();
		void redo();
		void clean();
		void as_rubber();
		void as_brush();
		void input(const ExMessage& msg) override;
	};
}

#endif // !UI_FORM_HPP
