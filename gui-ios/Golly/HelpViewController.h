// This file is part of Golly.
// See docs/License.html for the copyright notice.

#import <UIKit/UIKit.h>
#import <WebKit/WebKit.h>

// This is the view controller for the Help tab.

@interface HelpViewController : UIViewController <WKNavigationDelegate>
{
    IBOutlet WKWebView *htmlView;
    IBOutlet UIBarButtonItem *backButton;
    IBOutlet UIBarButtonItem *nextButton;
    IBOutlet UIBarButtonItem *contentsButton;
}

- (IBAction)doBack:(id)sender;
- (IBAction)doNext:(id)sender;
- (IBAction)doContents:(id)sender;

@end

// display given HTML file
void ShowHelp(const char* filepath);
