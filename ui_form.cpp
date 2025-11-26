#include "ui_form.hpp"

namespace chr {
	ui_base::ui_base(POINT position, int width, int height, COLORREF fill, COLORREF line)
		:pos_(position),
		width_(width), height_(height),
		fill_(fill), line_(line) {
	}
	void ui_base::basic_draw() const {
		setlinecolor(line_);
		setfillcolor(fill_);
		fillrectangle(pos_.x, pos_.y, pos_.x + width_, pos_.y + height_);
	}
	canvas::canvas(POINT position, int width, int height, COLORREF fill, COLORREF line)
		:ui_base(position, width, height, fill, line),
		is_rubber_(0), is_lbutton_(0),
		brush_(0) {
		IMAGE temp(width_ - 2, height_ - 2);
		auto ptr = GetImageBuffer(&temp);
		for (int i = 0; i < width_ - 2; i++) {
			for (int j = 0; j < height_ - 2; j++) {
				ptr[i + j * (width_ - 2)] = fill_;
			}
		}
		buffer_.push_front(std::move(temp));
	}
	void canvas::set_image(const IMAGE& image) {
		IMAGE copy = image;
		IMAGE temp(width_ - 2, height_ - 2);
		auto ptr = GetImageBuffer(&temp);
		auto src = GetImageBuffer(&copy);
		int m = (copy.getwidth() - width_ + 2) / 2;
		int n = (copy.getheight() - height_ + 2) / 2;
		for (int i = 0; i < width_ - 2; i++) {
			for (int j = 0; j < height_ - 2; j++) {
				if (m + i < copy.getwidth() && n + j < copy.getheight()) {
					ptr[i + j * (width_ - 2)] = src[m + i + (n + j) * (width_ - 2)];
				}
				else {
					ptr[i + j * (width_ - 2)] = fill_;
				}
			}
		}
		buffer_.push_front(std::move(temp));
		if (buffer_.size() > 64) {
			buffer_.pop_back();
		}
	}
	void canvas::draw() {
		basic_draw();
		putimage(pos_.x + 1, pos_.y + 1, &buffer_.front());
	}
	void canvas::undo() {
		if (buffer_.size() > 1) {
			IMAGE img = std::move(buffer_.front());
			buffer_.pop_front();
			undo_buffer_.push_front(std::move(img));
			if (undo_buffer_.size() > 64) {
				undo_buffer_.pop_back();
			}
		}
	}
	void canvas::redo() {
		if (undo_buffer_.size() > 0) {
			IMAGE img = std::move(undo_buffer_.front());
			undo_buffer_.pop_front();
			buffer_.push_front(std::move(img));
			if (buffer_.size() > 64) {
				buffer_.pop_back();
			}
		}
	}
	void canvas::clean() {
		IMAGE temp(width_ - 2, height_ - 2);
		auto ptr = GetImageBuffer(&temp);
		for (int i = 0; i < width_ - 2; i++) {
			for (int j = 0; j < height_ - 2; j++) {
				ptr[i + j * (width_ - 2)] = fill_;
			}
		}
		buffer_.push_front(std::move(temp));
		if (buffer_.size() > 64) {
			buffer_.pop_back();
		}
	}
	void canvas::as_rubber() {
		is_rubber_ = 1;
		is_lbutton_ = 0;
	}
	void canvas::as_brush() {
		is_rubber_ = 0;
		is_lbutton_ = 0;
	}
	void canvas::input(const ExMessage& msg) {
		if (msg.x >= pos_.x + 5 && msg.x < pos_.x + width_ - 5 &&
			msg.y >= pos_.y + 5 && msg.y < pos_.y + height_ - 5) {
			if (msg.message == WM_LBUTTONDOWN) {
				is_lbutton_ = 1;
				buffer_.push_front(buffer_.front());
				if (buffer_.size() > 64) {
					buffer_.pop_back();
				}
			}
			else if (msg.message == WM_LBUTTONUP) {
				is_lbutton_ = 0;
			}
			if (is_lbutton_) {
				if (is_rubber_) {
					SetWorkingImage(&buffer_.front());
					setfillcolor(fill_);
					solidcircle(msg.x - pos_.x - 1, msg.y - pos_.y - 1, 4);
					SetWorkingImage();
				}
				else {
					SetWorkingImage(&buffer_.front());
					setfillcolor(brush_);
					solidcircle(msg.x - pos_.x - 1, msg.y - pos_.y - 1, 4);
					SetWorkingImage();
				}
			}
		}
	}
	button::button(POINT position, int width, int height, COLORREF fill, COLORREF line)
		:ui_base(position, width, height, fill, line), normal_color_(fill), func_(nullptr),
		text_(L"button"), text_color_(0),
		is_hovered_(0), hover_color_(0xE0E0E0),
		is_pressed_(0), press_color_(0xD0D0D0) {
	}
	void button::set_colors(COLORREF normal, COLORREF hover, COLORREF press) {
		normal_color_ = normal;
		hover_color_ = hover;
		press_color_ = press;
		fill_ = normal_color_;
	}
	void button::draw() {
		if (is_pressed_) {
			fill_ = press_color_;
		}
		else if (is_hovered_) {
			fill_ = hover_color_;
		}
		else {
			fill_ = normal_color_;
		}
		basic_draw();
		settextcolor(text_color_);
		setbkmode(TRANSPARENT);
		settextstyle(height_ / 3, 0, L"Times New Roman");
		int text_width = textwidth(text_.c_str());
		int text_height = textheight(text_.c_str());
		int text_x = pos_.x + (width_ - text_width) / 2;
		int text_y = pos_.y + (height_ - text_height) / 2;
		outtextxy(text_x, text_y, text_.c_str());
	}
	void button::input(const ExMessage& msg) {
		if (msg.message == WM_MOUSEMOVE) {
			is_hovered_ = (msg.x >= pos_.x && msg.x < pos_.x + width_ &&
				msg.y >= pos_.y && msg.y < pos_.y + height_);
		}
		else if (msg.message == WM_LBUTTONDOWN) {
			if (is_hovered_) {
				is_pressed_ = true;
			}
		}
		else if (msg.message == WM_LBUTTONUP) {
			if (is_pressed_ && is_hovered_) {
				if (func_) {
					func_();
				}
			}
			is_pressed_ = false;
		}
	}
}