// This file is part of Golly.
// See docs/License.html for the copyright notice.

#import <UIKit/UIKit.h>
#import <WebKit/WebKit.h>

// This is the view controller for the Open tab.

@interface OpenViewController : UIViewController <UITableViewDelegate, UITableViewDataSource, WKNavigationDelegate>
{
    IBOutlet UITableView *optionTable;
    IBOutlet WKWebView *htmlView;
}

@end

// if any files exist in the Documents folder (created by iTunes file sharing)
// then move them into Documents/Rules/ if their names end with
// .rule/tree/table, otherwise assume they are patterns
// and move them into Documents/Saved/
void MoveSharedFiles();
